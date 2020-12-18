#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>

using namespace cv;
using namespace dnn;
using namespace std;

/// Darknetのモデル
String cfg = "C:/Users/ling/Downloads/mobilenetv2.cfg";
String weights = "C:/Users/ling/Downloads/mobilenetv2.weights";
std::vector<std::string> classes;
vector<string> coconame;

/// 推論結果のヒートマップを解析して表示
void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net);
void getStrFromText(string filename, vector<string>& vstr);


static float confThreshold = 0.5f;
static float nmsThreshold = 0.1f;

double scl = 1;
int a = 0;
int main()
{
    getStrFromText("coco.names.txt",coconame);

    VideoCapture camera;
    if (camera.open("C:/Users/ling/Downloads/a.mp4"))
    //if (camera.open(0))
    {
        /// YOLOv3のモデルを読み込む
        Net net = readNet(cfg, weights);
        Mat image, blob;
        std::vector<Mat> outs;
        std::vector<String> outNames = net.getUnconnectedOutLayersNames();
        for (bool loop = true; loop;)
        {
            camera >> image;
            
            
            Size imgsize = Size((image.size[1] - (image.size[1] % 416))* scl, ((image.size[0]) - (image.size[0] % 416))* scl);
            //Size imgsize = Size(320 * 2, 320 * 1);
            resize(image, image,imgsize);

            /// 画像をBLOBに変換して推論
            blobFromImage(image, blob, 1/255.0f, imgsize,true);
            net.setInput(blob,"");
            net.forward(outs, outNames);

            /// 推論結果をimageに描画
            postprocess(image, outs, net);

            //namedWindow("Image", WINDOW_NORMAL);
            //imshow("Image", image);
            imwrite("D:/box/"+to_string(a)+".jpeg",image);
            a++;
            cout << a << endl;
            //waitKey(1);
        }
        camera.release();
    }
}


void getStrFromText(string filename, vector<string>& vstr)
{
    ifstream ifs(filename);

    if (!ifs)
    {
        return;
    }

    string tmp;
    while (getline(ifs, tmp))
        vstr.push_back(tmp);
}


void drawPred(string cocname,int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));
    std::string label = cocname + format("%.2f", conf);
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
        Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.6, Scalar());
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            float* data = (float*)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;
                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }
    }
    if (outLayers.size() > 1 || (outLayerType == "Region" ))
    {
        std::map<int, std::vector<size_t> > class2indices;
        for (size_t i = 0; i < classIds.size(); i++)
        {
            if (confidences[i] >= confThreshold)
            {
                class2indices[classIds[i]].push_back(i);
            }
        }
        std::vector<Rect> nmsBoxes;
        std::vector<float> nmsConfidences;
        std::vector<int> nmsClassIds;
        for (std::map<int, std::vector<size_t> >::iterator it = class2indices.begin(); it != class2indices.end(); ++it)
        {
            std::vector<Rect> localBoxes;
            std::vector<float> localConfidences;
            std::vector<size_t> classIndices = it->second;
            for (size_t i = 0; i < classIndices.size(); i++)
            {
                localBoxes.push_back(boxes[classIndices[i]]);
                localConfidences.push_back(confidences[classIndices[i]]);
            }
            std::vector<int> nmsIndices;
            cv::dnn::NMSBoxes(localBoxes, localConfidences, confThreshold, nmsThreshold, nmsIndices);
            for (size_t i = 0; i < nmsIndices.size(); i++)
            {
                size_t idx = nmsIndices[i];
                nmsBoxes.push_back(localBoxes[idx]);
                nmsConfidences.push_back(localConfidences[idx]);
                nmsClassIds.push_back(it->first);
            }
        }
        boxes = nmsBoxes;
        classIds = nmsClassIds;
        confidences = nmsConfidences;
    }
    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        Rect box = boxes[idx];
        drawPred(coconame[classIds[idx]],classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);
    }
}