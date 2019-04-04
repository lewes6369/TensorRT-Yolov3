#include <opencv2/opencv.hpp>
#include "TrtNet.h"
#include "argsParser.h"
#include "configs.h"
#include <chrono>
#include "YoloLayer.h"
#include "dataReader.h"
#include "eval.h"
#include "pthread.h"

using namespace std;
using namespace argsParser;
using namespace Tn;
using namespace Yolo;


static cv::VideoCapture cap;  //静态全局变量仅对当前文件可见，其他文件不可访问
cv::Mat frame;
cv::Mat frame_;
vector<float> inputData;
vector<float> inputData_;
int outputCount;

vector<string> ClassName;
int classNum;
trtNet net;
int flag_exit=0;

vector<float> prepareImage(cv::Mat& img)
{
	using namespace cv;

	int c = parser::getIntValue("C");
	int h = parser::getIntValue("H");   //net h
	int w = parser::getIntValue("W");   //net w

	float scale = min(float(w) / img.cols, float(h) / img.rows);
	auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);

	cv::Mat rgb;
	cv::cvtColor(img, rgb, CV_BGR2RGB);
	cv::Mat resized;
	cv::resize(rgb, resized, scaleSize, 0, 0, INTER_CUBIC);

	cv::Mat cropped(h, w, CV_8UC3, 127);
	Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
	resized.copyTo(cropped(rect));

	cv::Mat img_float;
	if (c == 3)
		cropped.convertTo(img_float, CV_32FC3, 1 / 255.0);
	else
		cropped.convertTo(img_float, CV_32FC1, 1 / 255.0);

	//HWC TO CHW
	vector<Mat> input_channels(c);
	cv::split(img_float, input_channels);//通道分割

	vector<float> result(h*w*c);
	auto data = result.data();
	int channelLength = h * w;
	for (int i = 0; i < c; ++i) {
		memcpy(data, input_channels[i].data, channelLength * sizeof(float));
		data += channelLength;
	}

	return result;
}

void DoNms(vector<Detection>& detections, int classes, float nmsThresh)
{
	auto t_start = chrono::high_resolution_clock::now();

	vector<vector<Detection>> resClass;
	resClass.resize(classes);

	for (const auto& item : detections)
		resClass[item.classId].push_back(item);

	auto iouCompute = [](float * lbox, float* rbox)
	{
		float interBox[] = {
			max(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
			min(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
			max(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
			min(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
		};

		if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
			return 0.0f;

		float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
		return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
	};

	vector<Detection> result;
	for (int i = 0; i < classes; ++i)
	{
		auto& dets = resClass[i];
		if (dets.size() == 0)
			continue;

		sort(dets.begin(), dets.end(), [=](const Detection& left, const Detection& right) {
			return left.prob > right.prob;
		});

		for (unsigned int m = 0; m < dets.size(); ++m)
		{
			auto& item = dets[m];
			result.push_back(item);
			for (unsigned int n = m + 1; n < dets.size(); ++n)
			{
				if (iouCompute(item.bbox, dets[n].bbox) > nmsThresh)
				{
					dets.erase(dets.begin() + n);
					--n;
				}
			}
		}
	}

	//swap(detections,result);
	detections = move(result);

	auto t_end = chrono::high_resolution_clock::now();
	float total = chrono::duration<float, milli>(t_end - t_start).count();
	cout << "Time taken for nms is " << total << " ms." << endl;
}


vector<Bbox> postProcessImg(cv::Mat& img, vector<Detection>& detections, int classes)
{
	using namespace cv;

	int h = parser::getIntValue("H");   //net h
	int w = parser::getIntValue("W");   //net w

	//scale bbox to img
	int width = img.cols;
	int height = img.rows;
	float scale = min(float(w) / width, float(h) / height);
	float scaleSize[] = { width * scale,height * scale };

	//correct box
	for (auto& item : detections)
	{
		auto& bbox = item.bbox;
		bbox[0] = (bbox[0] * w - (w - scaleSize[0]) / 2.f) / scaleSize[0];
		bbox[1] = (bbox[1] * h - (h - scaleSize[1]) / 2.f) / scaleSize[1];
		bbox[2] /= scaleSize[0];
		bbox[3] /= scaleSize[1];
	}

	//nms
	float nmsThresh = parser::getFloatValue("nms");
	if (nmsThresh > 0)
		DoNms(detections, classes, nmsThresh);

	vector<Bbox> boxes;
	for (const auto& item : detections)
	{
		auto& b = item.bbox;
		Bbox bbox =
		{
			item.classId,   //classId
			max(int((b[0] - b[2] / 2.)*width),0), //left
			min(int((b[0] + b[2] / 2.)*width),width), //right
			max(int((b[1] - b[3] / 2.)*height),0), //top
			min(int((b[1] + b[3] / 2.)*height),height), //bot
			item.prob       //score
		};
		boxes.push_back(bbox);
	}

	return boxes;
}

vector<string> split(const string& str, char delim)
{
	stringstream ss(str);
	string token;
	vector<string> container;
	while (getline(ss, token, delim)) {
		container.push_back(token);
	}

	return container;
}


void error(const char *s)
{
	perror(s);
	assert(0);
	exit(EXIT_FAILURE);
}


void *fetch_in_thread(void *ptr)
{
	//in = get_image_from_stream(cap);
	cap >> frame;
	if (!frame.data)
	{
		flag_exit = 1;
		return 0;
	}
	//cv::Mat img = cv::imread(filename);
	inputData = prepareImage(frame);
	//in_s = resize_image(in, net.w, net.h);

	return 0;
}

void *detect_in_thread(void *ptr)
{

	//cout << frame.flags << endl;
	unique_ptr<float[]> outputData(new float[outputCount]);
	net.doInference(inputData_.data(), outputData.get());

	//Get Output    
	auto output = outputData.get();

	//first detect count
	int count = output[0];
	//later detect result
	vector<Detection> result;
	result.resize(count);
	memcpy(result.data(), &output[1], count * sizeof(Detection));

	auto boxes = postProcessImg(frame_, result, classNum);
	//outputs.emplace_back(boxes);

	//auto bbox = *outputs.begin();
	for (const auto& item : boxes)
	{	
		cv::rectangle(frame_, cv::Point(item.left, item.top), cv::Point(item.right, item.bot), cv::Scalar(0, 0, 255), 3, 8, 0);
		cv::rectangle(frame_, cv::Point(item.left, item.top - 20), cv::Point(item.right, item.top), cv::Scalar(0, 0, 255), CV_FILLED, 8, 0);
		cv::putText(frame_, ClassName[item.classId], cv::Point(item.left, item.top), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 255), 2);
		cout << "class=" << ClassName[item.classId] << " prob=" << item.score * 100 << endl;
		//cout << "left=" << item.left << " right=" << item.right << " top=" << item.top << " bot=" << item.bot << endl;
	}

	return 0;
}


void do_video_or_cam()
{
	if (!cap.isOpened()) {
		std::cout << "Error: video-stream can't be opened! \n";
		exit(EXIT_FAILURE);
	}
	pthread_t fetch_thread;
	pthread_t detect_thread;

	fetch_in_thread(0);
	inputData_ = inputData;
	frame_ = frame;
	detect_in_thread(0);
	fetch_in_thread(0);
	inputData_ = inputData;
	frame_ = frame;


	if (parser::getIntValue("display"))
	{
		cv::namedWindow("result", CV_WINDOW_NORMAL);
		cv::resizeWindow("result", 640, 480);
	}


	float fps = 0;

	//for (const auto& filename :fileNames)

	while (1)
	{
		auto t_start = std::chrono::high_resolution_clock::now();
		if (pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
		if (pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");//创造一个线程运行网络


		pthread_join(fetch_thread, 0);//塞入线程
		pthread_join(detect_thread, 0);



		if (parser::getIntValue("display"))
		{
			cv::imshow("result", frame_);
			if (cv::waitKey(10) == 27)
			{
				break;
			}
		}
		inputData_ = inputData;
		frame_ = frame;
		if (flag_exit == 1)
		{
			break;
		}
		

		auto t_end = std::chrono::high_resolution_clock::now();
		float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
		float cout = 1000. / total;
		fps = 0.9*fps + 0.1*cout;
		std::cout << "fps is " << fps << std::endl;
		
	}
	std::cout << "video stream close\n " << std::endl;
	cv::destroyAllWindows();
	cap.release();
}



int main(int argc, char* argv[])
{
	parser::ADD_ARG_STRING("prototxt", Desc("input yolov3 deploy"), DefaultValue(INPUT_PROTOTXT), ValueDesc("file"));
	parser::ADD_ARG_STRING("caffemodel", Desc("input yolov3 caffemodel"), DefaultValue(INPUT_CAFFEMODEL), ValueDesc("file"));
	parser::ADD_ARG_INT("C", Desc("channel"), DefaultValue(to_string(INPUT_CHANNEL)));
	parser::ADD_ARG_INT("H", Desc("height"), DefaultValue(to_string(INPUT_HEIGHT)));
	parser::ADD_ARG_INT("W", Desc("width"), DefaultValue(to_string(INPUT_WIDTH)));
	parser::ADD_ARG_STRING("calib", Desc("calibration image List"), DefaultValue(CALIBRATION_LIST), ValueDesc("file"));
	parser::ADD_ARG_STRING("mode", Desc("runtime mode"), DefaultValue(MODE), ValueDesc("fp32/fp16/int8"));
	parser::ADD_ARG_STRING("outputs", Desc("output nodes name"), DefaultValue(OUTPUTS));
	parser::ADD_ARG_INT("class", Desc("num of classes"), DefaultValue(to_string(DETECT_CLASSES)));
	parser::ADD_ARG_FLOAT("nms", Desc("non-maximum suppression value"), DefaultValue(to_string(NMS_THRESH)));
	parser::ADD_ARG_FLOAT("classname", Desc("class name"), DefaultValue(CLASS_NAME), ValueDesc("file"));
	parser::ADD_ARG_INT("display", Desc("whether display video"), DefaultValue(to_string(DISPLAY)));
	//input
	parser::ADD_ARG_STRING("inputstream", Desc("input stream"), DefaultValue(INPUT_STREAM));
	parser::ADD_ARG_INT("cam", Desc("cam"), DefaultValue(to_string(CAM)));
	parser::ADD_ARG_STRING("videofile", Desc("videofile"), DefaultValue(VIDEOFILE), ValueDesc("file"));
	//parser::ADD_ARG_STRING("input", Desc("input image file"), DefaultValue(INPUT_IMAGE), ValueDesc("file"));
	parser::ADD_ARG_STRING("evallist", Desc("eval gt list"), DefaultValue(EVAL_LIST), ValueDesc("file"));

	if (argc < 2) {
		parser::printDesc();
		return 1;
	}

	parser::parseArgs(argc, argv);

	string deployFile = parser::getStringValue("prototxt");
	string caffemodelFile = parser::getStringValue("caffemodel");

	vector<vector<float>> calibData;
	string calibFileList = parser::getStringValue("calib");  //校验文件的名字 txt
	string mode = parser::getStringValue("mode");
	if (calibFileList.length() > 0 && mode == "int8")
	{
		cout << "find calibration file,loading ..." << endl;

		ifstream file(calibFileList);
		if (!file.is_open())
		{
			cout << "read file list error,please check file :" << calibFileList << endl;
			return 1;
		}

		string strLine;
		while (getline(file, strLine))
		{
			cv::Mat img = cv::imread(strLine);
			auto data = prepareImage(img);
			calibData.emplace_back(data);
		}
		file.close();
	}

	RUN_MODE run_mode = RUN_MODE::FLOAT32;
	if (mode == "int8")
	{
		if (calibFileList.length() == 0)
			cout << "run int8 please input calibration file, will run in fp32" << endl;
		else
			run_mode = RUN_MODE::INT8;
	}
	else if (mode == "fp16")
	{
		run_mode = RUN_MODE::FLOAT16;
	}

	string outputNodes = parser::getStringValue("outputs");
	auto outputNames = split(outputNodes, ',');

	//can load from file
	string saveName = "yolov3_" + mode + ".engine";

	//#define LOAD_FROM_ENGINE
	std::ifstream serialize_iutput_stream(saveName, std::ios::in | std::ios::binary);
	net.set_mode(run_mode);

	if (!serialize_iutput_stream)
	{
		net.trtNet_engine(deployFile, caffemodelFile, outputNames, calibData, run_mode);
		cout << "save Engine..." << saveName << endl;
		net.saveEngine(saveName);
	}
	else
	{
		net.trtNet_engine(saveName);
	}

	outputCount = net.getOutputSize() / sizeof(float);
	//outputData(new float[outputCount]);
	//string listFile = parser::getStringValue("evallist");
	//list<string> fileNames;
	//list<vector<Bbox>> groundTruth;
	//if(listFile.length() > 0)
	//{
	//   std::cout << "loading from eval list " << listFile << std::endl; 
	//   tie(fileNames,groundTruth) = readObjectLabelFileList(listFile);
	//}
	//else
	//{
	//	string inputFileName = parser::getStringValue("input");
	//	fileNames.push_back(inputFileName);
		//该函数将一个新的元素加到vector的最后面，位置为当前最后一个元素的下一个元素，新的元素的值是val的拷贝（或者是移动拷贝）
	//}

	//list<vector<Bbox>> outputs;
	classNum = parser::getIntValue("class");
	string classFile = parser::getStringValue("classname");
	ifstream ClassNamelist(classFile);
	if (!ClassNamelist.is_open())
	{
		std::cout << "read file list error,please check file :" << classFile << std::endl;
		return 1;
	}
	string strLine;
	while (getline(ClassNamelist, strLine))
	{
		ClassName.push_back(strLine);
	}

	ClassNamelist.close();

	string inputstream = parser::getStringValue("inputstream");

	if (!inputstream.compare("video"))
	{
		string video_file = parser::getStringValue("videofile");
		cap.open(video_file);
		do_video_or_cam();
	}
	else if(!inputstream.compare("cam"))
	{
		int cam_index = parser::getIntValue("cam");
		cap.open(cam_index);
		do_video_or_cam();
	}
	return 0;
}
