#include "facedatabase.h"
#include "process.h"
#include <QDir>
#include <QFile>
#include <QXmlStreamWriter>
#include <QXmlStreamReader>
#include <QDateTime>
#include <cmath>

FaceDatabase::FaceDatabase() {
  loadXml("facedatabase.xml");
}

FaceDatabase::~FaceDatabase() {
}

void FaceDatabase::extractFeatures(const cv::Mat& faceMat, cv::Mat& featureVector) {
  if (faceMat.empty()) {
    featureVector = cv::Mat();
    return;
  }

  // 1. Standardize face image resolution to 64x64
  cv::Mat resized;
  cv::resize(faceMat, resized, cv::Size(64, 64));

  // 2. Convert to grayscale
  cv::Mat gray;
  if (resized.channels() == 3) {
    cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
  } else if (resized.channels() == 4) {
    cv::cvtColor(resized, gray, cv::COLOR_BGRA2GRAY);
  } else {
    gray = resized.clone();
  }

  // Equalize histogram to normalize lighting conditions
  cv::equalizeHist(gray, gray);

  // 3. Compute LBP feature vector
  cv::Mat lbpMat;
  process::computeLBP(gray, lbpMat);

  // 4. Compute CSLTP feature vector
  cv::Mat csltpMat;
  process::computeCSLTP(gray, csltpMat, 20);

  // 5. Compute 32-bin normalized grayscale histogram
  int histSize = 32;
  float range[] = { 0, 256 };
  const float* histRange = { range };
  cv::Mat hist;
  cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
  cv::Mat histMat = hist.reshape(1, 1); // 1x32 CV_32F

  // 6. Concatenate feature representations (LBP + CSLTP + Histogram)
  cv::Mat concatenated;
  cv::Mat tempMat;
  cv::hconcat(lbpMat, csltpMat, tempMat);
  cv::hconcat(tempMat, histMat, concatenated);

  // Ensure CV_32FC1 type
  concatenated.convertTo(featureVector, CV_32FC1);

  // 7. L2-normalize the combined feature vector
  double normVal = cv::norm(featureVector, cv::NORM_L2);
  if (normVal > 1e-6) {
    featureVector /= normVal;
  }
}

double FaceDatabase::computeCosineSimilarity(const cv::Mat& v1, const cv::Mat& v2) {
  if (v1.empty() || v2.empty() || v1.cols != v2.cols) {
    return 0.0;
  }

  double dot = v1.dot(v2);
  double norm1 = cv::norm(v1, cv::NORM_L2);
  double norm2 = cv::norm(v2, cv::NORM_L2);

  if (norm1 < 1e-6 || norm2 < 1e-6) return 0.0;

  double similarity = dot / (norm1 * norm2);
  return std::max(0.0, std::min(1.0, similarity));
}

bool FaceDatabase::addFace(const QString& personName, const cv::Mat& faceMat, const QString& dbDir) {
  if (personName.trimmed().isEmpty() || faceMat.empty()) {
    return false;
  }

  // Create directory for database face images if needed
  QDir dir(dbDir);
  if (!dir.exists()) {
    dir.mkpath(".");
  }

  // Save image file
  QString timestamp = QString::number(QDateTime::currentMSecsSinceEpoch());
  QString filename = personName + "_" + timestamp + ".jpg";
  QString filePath = dbDir + "/" + filename;
  cv::imwrite(filePath.toStdString(), faceMat);

  // Extract feature vector
  cv::Mat feat;
  extractFeatures(faceMat, feat);

  // Create record
  FaceRecord record;
  record.id = nextId++;
  record.name = personName.trimmed();
  record.imagePath = filePath;
  record.featureVector = feat;

  records.push_back(record);

  // Save updated database to XML
  saveXml();
  return true;
}

bool FaceDatabase::removeFaceRecord(int id) {
  for (auto it = records.begin(); it != records.end(); ++it) {
    if (it->id == id) {
      records.erase(it);
      saveXml();
      return true;
    }
  }
  return false;
}

bool FaceDatabase::removePerson(const QString& personName) {
  bool removed = false;
  for (auto it = records.begin(); it != records.end(); ) {
    if (it->name.compare(personName.trimmed(), Qt::CaseInsensitive) == 0) {
      it = records.erase(it);
      removed = true;
    } else {
      ++it;
    }
  }
  if (removed) {
    saveXml();
  }
  return removed;
}

void FaceDatabase::clear() {
  records.clear();
  saveXml();
}

bool FaceDatabase::compareFace(const cv::Mat& faceMat, QString& matchedName, double& similarityPercent, double minSimilarityThreshold) {
  if (faceMat.empty() || records.empty()) {
    matchedName = "Unknown";
    similarityPercent = 0.0;
    return false;
  }

  cv::Mat queryFeat;
  extractFeatures(faceMat, queryFeat);

  double maxSim = 0.0;
  QString bestMatch = "Unknown";

  for (const auto& rec : records) {
    double sim = computeCosineSimilarity(queryFeat, rec.featureVector);
    if (sim > maxSim) {
      maxSim = sim;
      bestMatch = rec.name;
    }
  }

  similarityPercent = maxSim * 100.0;

  if (similarityPercent >= minSimilarityThreshold) {
    matchedName = bestMatch;
    return true;
  } else {
    matchedName = "Unknown";
    return false;
  }
}

bool FaceDatabase::saveXml(const QString& xmlPath) {
  QFile file(xmlPath);
  if (!file.open(QIODevice::WriteOnly)) {
    return false;
  }

  QXmlStreamWriter writer(&file);
  writer.setAutoFormatting(true);
  writer.writeStartDocument();
  writer.writeStartElement("FaceDatabase");

  for (const auto& rec : records) {
    writer.writeStartElement("Record");
    writer.writeTextElement("ID", QString::number(rec.id));
    writer.writeTextElement("Name", rec.name);
    writer.writeTextElement("ImagePath", rec.imagePath);

    // Write feature vector as space-separated floats
    QString featStr;
    if (!rec.featureVector.empty()) {
      for (int c = 0; c < rec.featureVector.cols; ++c) {
        featStr += QString::number(rec.featureVector.at<float>(0, c)) + " ";
      }
    }
    writer.writeTextElement("Features", featStr.trimmed());

    writer.writeEndElement(); // Record
  }

  writer.writeEndElement(); // FaceDatabase
  writer.writeEndDocument();
  return true;
}

bool FaceDatabase::loadXml(const QString& xmlPath) {
  QFile file(xmlPath);
  if (!file.open(QIODevice::ReadOnly)) {
    return false;
  }

  records.clear();
  int maxId = 0;

  QXmlStreamReader reader(&file);
  while (!reader.atEnd() && !reader.hasError()) {
    QXmlStreamReader::TokenType token = reader.readNext();
    if (token == QXmlStreamReader::StartElement && reader.name() == "Record") {
      FaceRecord rec;
      while (!(reader.tokenType() == QXmlStreamReader::EndElement && reader.name() == "Record")) {
        reader.readNext();
        if (reader.tokenType() == QXmlStreamReader::StartElement) {
          QString elemName = reader.name().toString();
          QString elemText = reader.readElementText();

          if (elemName == "ID") {
            rec.id = elemText.toInt();
            if (rec.id > maxId) maxId = rec.id;
          } else if (elemName == "Name") {
            rec.name = elemText;
          } else if (elemName == "ImagePath") {
            rec.imagePath = elemText;
          } else if (elemName == "Features") {
            QStringList vals = elemText.split(' ', QString::SkipEmptyParts);
            if (!vals.isEmpty()) {
              rec.featureVector = cv::Mat(1, vals.size(), CV_32FC1);
              for (int i = 0; i < vals.size(); ++i) {
                rec.featureVector.at<float>(0, i) = vals[i].toFloat();
              }
            }
          }
        }
      }
      records.push_back(rec);
    }
  }

  nextId = maxId + 1;
  return true;
}

std::vector<FaceRecord> FaceDatabase::getRecords() const {
  return records;
}

QStringList FaceDatabase::getEnrolledPersons() const {
  QStringList persons;
  for (const auto& rec : records) {
    if (!persons.contains(rec.name)) {
      persons.append(rec.name);
    }
  }
  return persons;
}

int FaceDatabase::recordCount() const {
  return static_cast<int>(records.size());
}
