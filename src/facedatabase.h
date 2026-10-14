#ifndef FACEDATABASE_H
#define FACEDATABASE_H

#include <QString>
#include <QStringList>
#include <QMap>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

struct FaceRecord {
  int id;
  QString name;
  QString imagePath;
  cv::Mat featureVector;
};

class FaceDatabase {
 public:
  FaceDatabase();
  ~FaceDatabase();

  // Feature extraction
  static void extractFeatures(const cv::Mat& faceMat, cv::Mat& featureVector);
  static double computeCosineSimilarity(const cv::Mat& v1, const cv::Mat& v2);

  // Database operations
  bool addFace(const QString& personName, const cv::Mat& faceMat, const QString& dbDir = "database_faces");
  bool removeFaceRecord(int id);
  bool removePerson(const QString& personName);
  void clear();

  // Face comparison against existing database
  bool compareFace(const cv::Mat& faceMat, QString& matchedName, double& similarityPercent, double minSimilarityThreshold = 55.0);

  // Persistence
  bool saveXml(const QString& xmlPath = "facedatabase.xml");
  bool loadXml(const QString& xmlPath = "facedatabase.xml");

  // Getters
  std::vector<FaceRecord> getRecords() const;
  QStringList getEnrolledPersons() const;
  int recordCount() const;

 private:
  std::vector<FaceRecord> records;
  int nextId = 1;
};

#endif // FACEDATABASE_H
