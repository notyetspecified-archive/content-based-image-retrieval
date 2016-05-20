#include <stdio.h>
#include <tchar.h>
#include <opencv/cv.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include "ml.h"
#include <fstream>
#include <math.h>

using namespace cv;
using namespace std;

//Globals
#define IMG_PATH "../img/"
#define IMG_INDEX "/images.txt"
#define NUMBER_OF_WORDS 10000
#define NUMBER_OF_FILES_CAT 100
#define NUMBER_OF_TEST_FILES_CAT 50
#define NUMBER_OF_FILES_RETRIVED 10
#define DATABASE_FILENAME "../models/database.xml"
#define INDEXES_FILENAME "../models/indexes.xml"
#define CLASSIFIERS_FILENAME "../models/classifier.xml"
#define DESCRITORS_FILENAME "../models/descriptors.xml"
#define CONFUSION_MATRIX_FILENAME "../models/confusion_matrix.csv"
#define TEST_IMG_FILENAME "../img/test/test4.jpg"

bool debug = true;

vector<string> categories = { "aquarium", "bar", "botanical_garden", "street", "windmill" };
vector<vector<Mat>> featuresVector;
vector<vector<float>> descriptorsVector;
vector<float> occurrencesVector;
vector<vector<float>> occurrencesInImageVector;
vector<vector<float>> indexes;
Mat indexesMat;
Mat descriptorsMat;
CvSVM svm;
vector<vector<int>> confusionMatrix;
vector<vector<int>> tablesOfConfusion;
vector<int> averageTableOfConfusion;

Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SIFT");
Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");
BOWKMeansTrainer bowTrainer = BOWKMeansTrainer(NUMBER_OF_WORDS, TermCriteria(), 1, KMEANS_PP_CENTERS);
BOWImgDescriptorExtractor bowExtractor = BOWImgDescriptorExtractor(detector, matcher);

bool openImage(const std::string &filename, Mat &image);
void drawKeypoints(string windowName, Mat &image, std::vector<KeyPoint> &keypoints, std::vector<int> &words);
void detecLocalFeatures();
vector<string> getFileNames(string);
void gatherLocalFeatures();
void clusterFeatures();
void saveDescriptors();
void loadDescriptors();
void saveVocabulary();
void loadVocabulary();
void computeDescriptors();
void computeOccurrences();
void computeWeights();
void saveIndexes();
void loadIndexes();
void trainClassifier();
void loadClassifier();
vector<int> nearestNeighbors(Mat);
vector<int> nearestNeighborsCat(vector<int>, int);
vector<int> sortDescriptors(vector<float>);
bool contains(vector<int>, int);
void createConfusionMatrix();
char *string2char(string);
void saveConfusionMatrix();
void loadConfusionMatrix();
void showConfusionMatrix();
void createConfusionTable();
void showTablesOfConfusion();
void createAverageTableOfConfusion();
void showAverageTableOfConfusion();
void showStatistics();
vector<string> getTestFileNames(int);
bool existFile(string);
Mat getImage(int);

int main(int argc, char** argv)
{
	initModule_nonfree();
	FileStorage fs;

	if (!fs.open(DATABASE_FILENAME, FileStorage::READ))
	{
		/// Local Features Detection
		cout << "Detecting all Local Features" << endl;
		detecLocalFeatures();
		cout << "Done\n" << endl;

		/// Gather Features
		cout << "Gathering all Local Features" << endl;
		gatherLocalFeatures();
		cout << "Done\n" << endl;

		/// Cluster Features
		cout << "Clustering Features" << endl;
		clusterFeatures();
		cout << "Done\n" << endl;

		/// Write Vocabulary
		cout << "Saving Vocabulary" << endl;
		saveVocabulary();
		cout << "Done\n" << endl;
	}
	else
	{
		/// Load Vocabulary
		cout << "Loading Vocabulary" << endl;
		loadVocabulary();
		cout << "Done\n" << endl;
	}

	if (!fs.open(INDEXES_FILENAME, FileStorage::READ))
	{
		/// Compute Descriptors
		cout << "Computing Descriptors" << endl;
		computeDescriptors();
		cout << "Done\n" << endl;

		/// Save Descriptors
		cout << "Saving Descriptors" << endl;
		saveDescriptors();
		cout << "Done\n" << endl;

		/// Compute Occurrences
		cout << "Computing Occurrences" << endl;
		computeOccurrences();
		cout << "Done\n" << endl;

		/// Compute Weights
		cout << "Computing Weights" << endl;
		computeWeights();
		cout << "Done\n" << endl;

		/// Save Indexes
		cout << "Saving Indexes" << endl;
		saveIndexes();
		cout << "Done\n" << endl;
	}
	else
	{
		/// Load Descriptors
		cout << "Loading Descriptors" << endl;
		loadDescriptors();
		cout << "Done\n" << endl;

		/// Load Indexes
		cout << "Loading Indexes" << endl;
		loadIndexes();
		cout << "Done\n" << endl;
	}

	if (!fs.open(CLASSIFIERS_FILENAME, FileStorage::READ))
	{
		/// Train Classifier
		cout << "Training Classifier" << endl;
		trainClassifier();
		cout << "Done\n" << endl;
	}
	else
	{
		/// Load Classifier
		cout << "Loading Classifier" << endl;
		loadClassifier();
		cout << "Done\n" << endl;
	}

	if (!existFile(CONFUSION_MATRIX_FILENAME))
	{
		/// Create Confusion Matrix
		cout << "Creating Confusion Matrix" << endl;
		createConfusionMatrix();
		saveConfusionMatrix();
		cout << "Done\n" << endl;
	}
	else loadConfusionMatrix();

	/// Print Confusion Matrix
	showConfusionMatrix();
	cout << endl;

	/// Create Confusion Table
	cout << "Creating Confusion Table" << endl;
	createConfusionTable();
	cout << "Done\n" << endl;

	//showTablesOfConfusion();

	/// Create Average Table of Confusion
	cout << "Creating Average Table of Confusion" << endl;
	createAverageTableOfConfusion();
	cout << "Done\n" << endl;

	showAverageTableOfConfusion();
	showStatistics();

	cout << "\n\n\nFinding Category" << endl;
	Mat img;
	openImage(TEST_IMG_FILENAME, img);
	vector<KeyPoint> keypoints;
	detector->detect(img, keypoints);
	Mat descriptor;
	bowExtractor.compute(img, keypoints, descriptor);
	int catIndex = (int) svm.predict(descriptor);

	cout << "Image Category: " << categories[catIndex] << endl;
	cout << "Done\n" << endl;

	cout << "Calculating Nearest Neighbors" << endl;

	vector<int> neighbors = nearestNeighbors(descriptor);
	neighbors = nearestNeighborsCat(neighbors, catIndex);

	imshow("Original", img);

	for (int i = 0; i < NUMBER_OF_FILES_RETRIVED; i++)
	{
		String windowName = "Nearest Neighbor ";
		windowName.append(to_string(i));
		imshow(windowName, getImage(neighbors[i]));
	}

	cout << "Done\n" << endl;

	waitKey(0);
	return 0;
}

void detecLocalFeatures()
{
	for (unsigned int i = 0; i < categories.size(); i++)
	{
		vector<Mat> tempFeatures;
		string categoryName = categories[i];
		vector<string> fileNames = getFileNames(categoryName);
		int fileNumber = fileNames.size();

		if (debug)
			fileNumber = NUMBER_OF_FILES_CAT;

		for (int j = 0; j < fileNumber; j++)
		{
			Mat image, features;
			vector<KeyPoint> keypoints;

			openImage(fileNames[j], image);
			detector->detect(image, keypoints);
			extractor->compute(image, keypoints, features);
			tempFeatures.push_back(features);
		}

		featuresVector.push_back(tempFeatures);
	}
}

vector<string> getFileNames(string categoryName)
{
	string fileNameIndex = IMG_PATH + categoryName + IMG_INDEX;
	string line;
	string tempFileName;
	vector<string> fileNames;

	ifstream file(fileNameIndex);
	if (file.is_open())
	{
		while (getline(file, line))
		{
			tempFileName = IMG_PATH + categoryName + "/" + line;
			fileNames.push_back(tempFileName);
		}
		file.close();
	}

	return fileNames;
}

vector<string> getTestFileNames(int category)
{
	vector<string> fileNames = getFileNames(categories[category]);
	vector<string> catFileNames;

	int start = NUMBER_OF_FILES_CAT;
	int finish = NUMBER_OF_FILES_CAT + NUMBER_OF_TEST_FILES_CAT;

	for (int i = start; i < finish; i++)
		catFileNames.push_back(fileNames[i]);

	return catFileNames;
}

void gatherLocalFeatures()
{
	for (unsigned int i = 0; i < featuresVector.size(); i++)
	{
		for (unsigned int j = 0; j < featuresVector[i].size(); j++)
			bowTrainer.add(featuresVector[i][j]);
	}
}

void clusterFeatures()
{
	Mat dictionary = bowTrainer.cluster();
	bowExtractor.setVocabulary(dictionary);
}

void saveVocabulary()
{
	FileStorage fs(DATABASE_FILENAME, FileStorage::WRITE);
	fs << "Vocabulary" << bowExtractor.getVocabulary();
	fs.release();
}

void loadVocabulary()
{
	FileStorage fs(DATABASE_FILENAME, FileStorage::READ);
	Mat vocabulary;
	fs["Vocabulary"] >> vocabulary;
	bowExtractor.setVocabulary(vocabulary);
	fs.release();
}

void computeDescriptors()
{
	for (unsigned int i = 0; i < categories.size(); i++)
	{
		vector<float> tempDescriptors;
		string categoryName = categories[i];
		vector<string> fileNames = getFileNames(categoryName);
		int fileNumber = fileNames.size();

		if (debug)
			fileNumber = NUMBER_OF_FILES_CAT;

		for (int j = 0; j < fileNumber; j++)
		{
			Mat image, descriptors;
			vector<KeyPoint> keypoints;

			openImage(fileNames[j], image);
			detector->detect(image, keypoints);
			bowExtractor.compute(image, keypoints, descriptors);

			for (int k = 0; k < descriptors.cols; k++)
				tempDescriptors.push_back(descriptors.at<float>(0, k));

			descriptorsVector.push_back(tempDescriptors);
			tempDescriptors.clear();
		}
	}
}

void computeOccurrences()
{
	//Occurrences in database
	for (unsigned int i = 0; i < NUMBER_OF_WORDS; i++)
		occurrencesVector.push_back(0);

	//Occurrences in each Image
	for (unsigned int i = 0; i < descriptorsVector.size(); i++)
	{
		vector<float> temp;

		for (unsigned int j = 0; j < NUMBER_OF_WORDS; j++)
			temp.push_back(0.0);

		occurrencesInImageVector.push_back(temp);
	}


	for (unsigned int i = 0; i < descriptorsVector.size(); i++)
	{
		for (unsigned int j = 0; j < descriptorsVector[i].size(); j++)
		{
			occurrencesVector.at(j) += descriptorsVector[i][j];
			occurrencesInImageVector.at(i).at(j) += descriptorsVector[i][j];
		}
	}
}

void computeWeights()
{
	Mat voc = bowExtractor.getVocabulary();

	for (unsigned int i = 0; i < occurrencesInImageVector.size(); i++)
	{
		vector<float> temp;

		for (unsigned int j = 0; j < occurrencesInImageVector[i].size(); j++)
		{
			float weight = (float)occurrencesInImageVector[i][j] / NUMBER_OF_WORDS *
				log(occurrencesInImageVector.size() / occurrencesVector[j]);

			temp.push_back(weight + descriptorsVector[i][j]);
		}

		indexes.push_back(temp);
	}
}

void saveDescriptors()
{
	Mat temp(descriptorsVector.size(), NUMBER_OF_WORDS, CV_32F);

	for (int i = 0; i < temp.rows; i++)
		for (int j = 0; j < temp.cols; j++)
			temp.at<float>(i, j) = descriptorsVector.at(i).at(j);

	FileStorage fs(DESCRITORS_FILENAME, FileStorage::WRITE);
	fs << "Descriptors" << temp;
	fs.release();

	descriptorsMat = temp;
}

void loadDescriptors()
{
	FileStorage fs(DESCRITORS_FILENAME, FileStorage::READ);
	fs["Descriptors"] >> descriptorsMat;
	fs.release();
}

void saveIndexes()
{
	Mat temp(indexes.size(), NUMBER_OF_WORDS, CV_32F);

	for (int i = 0; i < temp.rows; i++)
		for (int j = 0; j < temp.cols; j++)
			temp.at<float>(i, j) = indexes.at(i).at(j);

	normalize(temp, temp);

	FileStorage fs(INDEXES_FILENAME, FileStorage::WRITE);
	fs << "Indexes" << temp;
	fs.release();

	indexesMat = temp;
}

void loadIndexes()
{
	FileStorage fs(INDEXES_FILENAME, FileStorage::READ);
	fs["Indexes"] >> indexesMat;
	fs.release();
}

void trainClassifier()
{
	CvSVMParams params;
	Mat labels(0, 1, CV_32F);
	int numberCategories = categories.size();
	int numberImages = indexesMat.rows;
	int imagesPerCategory = numberImages / numberCategories;
	int category = 0;
	int imgCounter = 0;

	//Adding category
	for (int i = 0; i < indexesMat.rows; i++)
	{
		labels.push_back(category);
		imgCounter++;

		if (imgCounter == imagesPerCategory)
		{
			category++;
			imgCounter = 0;
		}
	}

	//Training
	svm.train_auto(indexesMat, labels, Mat(), Mat(), params);
	svm.save(CLASSIFIERS_FILENAME);
}

void loadClassifier()
{
	svm.load(CLASSIFIERS_FILENAME);
}

vector<int> nearestNeighbors(Mat descriptor)
{
	vector<float> ranks;
	vector<int> rankDescriptors;
	float sumDQ = 0;
	float sumD2 = 0;
	float sumQ2 = 0;

	for (int i = 0; i < indexesMat.rows; i++)
	{
		sumDQ = 0;
		sumD2 = 0;
		sumQ2 = 0;

		for (int j = 0; j < indexesMat.cols; j++)
		{
			sumDQ += indexesMat.at<float>(i, j) * descriptor.at<float>(0, j);
			sumD2 += pow(indexesMat.at<float>(i, j), 2);
			sumQ2 += pow(descriptor.at<float>(0, j), 2);
		}

		/*
			sum(d(i) * q(i))
		---------------------------
		 sum(d(i)^2) * sum(q(i)^2)
		*/

		ranks.push_back(sumDQ / (sqrt(sumD2)*sqrt(sumQ2)));
	}

	rankDescriptors = sortDescriptors(ranks);

	return rankDescriptors;
}

vector<int> nearestNeighborsCat(vector<int> neighbors, int category)
{
	int start = NUMBER_OF_FILES_CAT*category;
	int finish = NUMBER_OF_FILES_CAT*(category+1);
	vector<int> neighborsCat;

	for (unsigned int i = 0; i < neighbors.size(); i++)
	{
		if (neighbors[i] >= start && neighbors[i] < finish)
			neighborsCat.push_back(neighbors[i]);
	}

	return neighborsCat;
}

vector<int> sortDescriptors(vector<float> ranks)
{
	float tempValue = 0;
	int tempIndex = 0;
	vector<int> sortedIndexes;

	while (sortedIndexes.size() < ranks.size())
	{
		tempValue = 0;

		for (unsigned int i = 0; i < ranks.size(); i++)
		{
			if (ranks[i] > tempValue && !contains(sortedIndexes, i))
			{
				tempValue = ranks[i];
				tempIndex = i;
			}
		}

		sortedIndexes.push_back(tempIndex);
	}

	return sortedIndexes;
}

bool contains(vector<int> array, int value)
{
	for (unsigned int i = 0; i < array.size(); i++)
	{
		if (array[i] == value)
			return true;
	}

	return false;
}

void createConfusionMatrix()
{
	for (unsigned int i = 0; i < categories.size(); i++)
	{
		vector<string> catFileNames = getTestFileNames(i);
		vector<int> categoryLine;

		for (unsigned int j = 0; j < categories.size(); j++)
			categoryLine.push_back(0);

		for (unsigned int j = 0; j < catFileNames.size(); j++)
		{
			Mat img, descriptor;
			openImage(catFileNames[j], img);
			vector<KeyPoint> keypoints;
			detector->detect(img, keypoints);
			bowExtractor.compute(img, keypoints, descriptor);
			int catIndex = (int)svm.predict(descriptor);
			categoryLine[catIndex]++;
		}

		for (unsigned int j = 0; j < categories.size(); j++)
			cout << categoryLine[j] << ",";

		cout << endl;

		confusionMatrix.push_back(categoryLine);
	}

	saveConfusionMatrix();
}

void saveConfusionMatrix()
{
	ofstream myfile(CONFUSION_MATRIX_FILENAME);
	
	if (myfile.is_open())
	{
		for (unsigned int i = 0; i < confusionMatrix.size(); i++)
		{
			for (unsigned int j = 0; j < confusionMatrix[i].size(); j++)
			{
				myfile << confusionMatrix[i][j];

				if (j + 1 != confusionMatrix[i].size())
					myfile << ",";
			}

			myfile << "\n";
		}
	}
}

void loadConfusionMatrix()
{
	ifstream file(CONFUSION_MATRIX_FILENAME);
	string line;
	char *pch;

	if (file.is_open())
	{
		while (getline(file, line))
		{
			vector<int> temp;
			pch = strtok(string2char(line), ",");

			while (pch != NULL)
			{
				temp.push_back(atoi(pch));
				pch = strtok(NULL, ",");
			}

			confusionMatrix.push_back(temp);
		}
		file.close();
	}

	file.close();
}

void showConfusionMatrix()
{
	cout << "Confusion Matrix" << endl;
	for (unsigned int i = 0; i < confusionMatrix.size(); i++)
	{
		for (unsigned int j = 0; j < confusionMatrix[i].size(); j++)
		{
			cout << confusionMatrix[i][j];
			if (j + 1 != confusionMatrix[i].size())
				cout << "\t";
		}

		cout << endl;
	}
}

void createConfusionTable()
{
	for (unsigned int i = 0; i < categories.size(); i++)
	{
		//TP,TN,FP,FN
		vector<int> tableOfConfusion;
		tableOfConfusion.push_back(0);
		tableOfConfusion.push_back(0);
		tableOfConfusion.push_back(0);
		tableOfConfusion.push_back(0);

		tablesOfConfusion.push_back(tableOfConfusion);
	}

	int total = 0;

	for (unsigned int i = 0; i < confusionMatrix.size(); i++)
	{
		//TP
		tablesOfConfusion[i][0] = confusionMatrix[i][i];

		for (unsigned int j = 0; j < confusionMatrix[i].size(); j++)
		{
			total += confusionMatrix[i][j];

			if (i != j)
			{
				//FP
				tablesOfConfusion[i][2] += confusionMatrix[i][j];
				//FN
				tablesOfConfusion[j][3] += confusionMatrix[i][j];
			}
		}
	}

	//TN = total - TP - FP - FN
	for (unsigned int i = 0; i < categories.size(); i++)
		tablesOfConfusion[i][1] = total - tablesOfConfusion[i][0] - tablesOfConfusion[i][2] - tablesOfConfusion[i][3];
}

void showTablesOfConfusion()
{
	for (unsigned int i = 0; i < tablesOfConfusion.size(); i++)
	{
		cout << categories[i] << endl;
		cout << tablesOfConfusion[i][0] << "\t" << tablesOfConfusion[i][2] << endl;
		cout << tablesOfConfusion[i][3] << "\t" << tablesOfConfusion[i][1] << endl;
		cout << endl;
	}
}

void createAverageTableOfConfusion()
{
	averageTableOfConfusion.push_back(0);
	averageTableOfConfusion.push_back(0);
	averageTableOfConfusion.push_back(0);
	averageTableOfConfusion.push_back(0);

	for (unsigned int i = 0; i < tablesOfConfusion.size(); i++)
	{
		averageTableOfConfusion[0] += tablesOfConfusion[i][0];
		averageTableOfConfusion[1] += tablesOfConfusion[i][1];
		averageTableOfConfusion[2] += tablesOfConfusion[i][2];
		averageTableOfConfusion[3] += tablesOfConfusion[i][3];
	}

	for (unsigned int i = 0; i < averageTableOfConfusion.size(); i++)
	{
		averageTableOfConfusion[i] = averageTableOfConfusion[i] / categories.size();
	}
}

void showAverageTableOfConfusion()
{
	cout << "Average" << endl;
	cout << averageTableOfConfusion[0] << "\t" << averageTableOfConfusion[2] << endl;
	cout << averageTableOfConfusion[3] << "\t" << averageTableOfConfusion[1] << endl;
	cout << endl;
}

void showStatistics()
{
	float TP = (float)averageTableOfConfusion[0];
	float TN = (float)averageTableOfConfusion[1];
	float FP = (float)averageTableOfConfusion[2];
	float FN = (float)averageTableOfConfusion[3];
	float P = TP + FN;
	float N = FP + TN;

	cout << "True Positive Rate (Sensitivity): " << TP / (TP + FN) << endl;
	cout << "True Negative Rate (Specificity): " << TN / (FP + TN) << endl;
	cout << "Positive Predictive Value (Precision): " << TP / (TP + FP) << endl;
	cout << "Negative Predictive Value: " << TN / (TN + FN) << endl;
	cout << "False Positive Rate (Fall-Out): " << FP / (FP + TN) << endl;
	cout << "False Discovery Rate: " << FP / (FP + TP) << endl;
	cout << "False Negative Rate (Miss Rate): " << FN / (FN + TP) << endl;
	cout << endl;
	cout << "Accuracy: " << (TP + TN) / (P + N) << endl;
	cout << "F1 score: " << (2*TP) / ((2*TP) + FP + FN) << endl;
	cout << "Matthews Correlation Coefficient: " << ((TP*TN) - (FP*FN)) / sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) << endl;
	cout << "Informedness: " << (TP / (TP + FN)) + (TN / (FP + TN)) - 1 << endl;
	cout << "Markedness: " << (TP / (TP + FP)) + (TN / (TN + FN)) - 1 << endl;
}

char *string2char(string str)
{
	char *a = new char[str.size() + 1];
	a[str.size()] = 0;
	memcpy(a, str.c_str(), str.size());

	return a;
}

bool existFile(string fileName)
{
	ifstream infile(fileName);
	return infile.good();
}

Mat getImage(int index)
{
	Mat image;
	int counter = 0;
	bool flag = false;

	for (unsigned int i = 0; i < categories.size(); i++)
	{
		string categoryName = categories[i];
		vector<string> fileNames = getFileNames(categoryName);
		int fileNumber = fileNames.size();

		if (debug)
			fileNumber = NUMBER_OF_FILES_CAT;

		for (int j = 0; j < fileNumber; j++)
		{
			if (counter == index)
			{
				openImage(fileNames[j], image);
				flag = true;
			}
			if (flag)
				break;

			counter++;
		}
		if (flag)
			break;
	}

	return image;
}

bool openImage(const std::string &filename, Mat &image)
{
	image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!image.data) {
		std::cout << " --(!) Error reading image " << filename << std::endl;
		return false;
	}
	return true;
}

