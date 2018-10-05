#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>


using namespace cv;
using namespace std;



const string face_cascade_name = "haarcascade_frontalface_alt.xml";
const string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
const string output_folder = "D:\\saveFace";

vector<Mat> images;
vector<int> labels;
Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
bool recognization = false;
bool faceReplace = false;
//Mat normImage = imread("D:\\attfacedatabase\\att_faces\\s1\\1.pgm", 0);

int countFace = 1;
int label[10] = {0,1,2,3,4,5,6,7,8,9};
string labelName[10] = {""};
int person = 0;
string personName[3] = { "stranger","stranger", "stranger", };
Mat lena ;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}
void setupWebCam(VideoCapture &videoCapture_P) {
	try {
		videoCapture_P.open(0);
	}
	catch (cv::Exception &e) {}
	if (!videoCapture_P.isOpened()) {
		cout << "fail to open webCam" << endl;
	}
	cout << "Loaded camera "  << endl;
}

void setupDetectors(CascadeClassifier &faceCCF_P, CascadeClassifier &eyeCCF_P) {
	try {
		faceCCF_P.load(face_cascade_name);
	}
	catch (cv::Exception &e) {
		
	}
	if (faceCCF_P.empty()) {
		cout << "facecacade is empty" << endl;
		exit(1);
	}
	cout << "loading faceCacade successfull" << endl;

	try
	{
		eyeCCF_P.load(eyes_cascade_name);

	}
	catch (cv::Exception &e)
	{	
	}
	if (eyeCCF_P.empty()) {
		exit(1);
	}
	cout << "loading eyeCacade successfull" << endl;
}
void detectAndDisplay(Mat frame, CascadeClassifier &faceCCF_P, CascadeClassifier &eyeCCF)
{
	

	std::vector<Rect> faces;
	Mat frame_gray;
	bool faceAndEyeDetec = false;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	faceCCF_P.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{

		if (recognization) {
			Mat clipFromVideo;
			Mat faceTest;
			Rect face = faces[i];
			clipFromVideo = frame_gray(face).clone();
			if (clipFromVideo.cols > 100) {
				resize(clipFromVideo, faceTest, Size(92, 112));
				int predictedLabel = model->predict(faceTest);
				for (int j= 0; j < 10; j++) {
					if (predictedLabel == label[j]) {
						//cout << "it's" << labelName[j] << endl;
						personName[i] = labelName[j];
						break;
						//putText(frame, labelName[i], Point(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
					}
					else
					{
						personName[i] = "stranger";
					}
				}
			}
		}
		if (!faceReplace)
		{
			Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
			ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
			putText(frame, personName[i], Point(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));

		}
		
		else {


			resize(lena, lena, Size(faces[i].width, faces[i].height));

			lena.copyTo(frame(cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height)));
		}
		
		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyeCCF.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		if (eyes.size() == 2) {
			faceAndEyeDetec = true;
		}

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
		}

		//putText(frame, "huangsuyu", Point(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
	}


	//1------press a to save recognized face into the face database 
	//2------press b to train the model 
	//3------press c to replace  human face by selected photo

	int switchButton =waitKey(23);
	switch ((char)switchButton) {

		//press a

	case 97:
		// ((char)switchButton == 97 && faceAndEyeDetec && countFace < 11) {


		//we save 10 pics of human face into database
		if (faceAndEyeDetec && countFace < 11) {
			try {

				Mat clip;
				Mat myFace;
				Rect face = faces[0];
				clip = frame_gray(face).clone();
				if (clip.cols > 100) {
					resize(clip, myFace, Size(92, 112));
					imwrite(format("%s\\s%d\\%d.pgm", output_folder.c_str(), label[person], countFace), myFace);
					cout << "save " << countFace << endl;
					ofstream outFile("D:\\heihei.txt", ios::app);
					outFile << "D:/saveFace" << "/s" << label[person] << "/" << countFace << ".pgm" << ";" << label[person] << "\n";

					if (countFace == 10) {
						cout << "save face complete" << endl;
						cout << "please input this person's name" << endl;
						cin >> labelName[person];
					}
					countFace++;
				}
			}
			catch (Exception e) {}
		}


		if (countFace == 11) {
			cout << labelName[person] << "faces have been saved !"<< endl;
			person++;
			countFace = 1;
		}
		break;
	case 98:

		cout << "start trainning " << endl;
		/*
		vector<Mat> images;
		vector<int> labels;
		Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
		*/
		read_csv("D:\\heihei.txt", images, labels);
		model->train(images, labels);
		

		recognization = true;

		/*
		Mat clipFromVideo;
		Mat faceTest;
		for (size_t i = 0; i < faces.size(); i++) {
			Rect face = faces[i];
			clipFromVideo = frame_gray(face).clone();
			if (clipFromVideo.cols > 100) {
				resize(clipFromVideo, faceTest, Size(92, 112));
				int predictedLabel = model->predict(faceTest);
				for (int i = 0; i < 10; i++) {
					if (predictedLabel == label[i]) {
						cout << "it's" << labelName[i] << endl;
						personName[i] = labelName[i];
						//putText(frame, labelName[i], Point(faces[i].x, faces[i].y), FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));
					}
					else
					{
						personName[i] = "stranger";
					}
				}
			}
		}
		*/
		break;
	case 99:
		faceReplace = !faceReplace;
		if (faceReplace) {
			cout << "FACE REPALCE ON" << endl;
		}
		else
		{
			cout << "FACE REPLACE OFF" << endl;
		}
		break;
	default:
		break;

		//-- Show what you got
	}
	cv::imshow("hi", frame);
	
}
void drawingFace(VideoCapture &videoCapture_P, CascadeClassifier &faceCCF_P, CascadeClassifier &eyeCCF) {
	Mat frame;

	while (true)
	{
		videoCapture_P >> frame;

		
		if (!frame.empty())
		{
			detectAndDisplay(frame,faceCCF_P,eyeCCF);
		}
		else
		{
			printf(" --(!) No captured frame -- Break!"); break;
		}

		int c = waitKey(10);
		if ((char)c == 27) { break; }

	}
}

int main(int argc, char* argv[]) {
	try {
		lena = imread("D:\\wuyanzu.jpg");
	}
	catch (Exception e) {

	}

	if (lena.empty()) {
		cout << "lena died" << endl;
	}
	CascadeClassifier faceCCf;
	CascadeClassifier eyeCCF;
	//cout << "nom rows:" << normImage.rows << endl;
	
	VideoCapture videoCapture;

	setupDetectors(faceCCf,eyeCCF);
	setupWebCam(videoCapture);

	drawingFace(videoCapture, faceCCf, eyeCCF);
}



