#include<opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

std::vector<Point>  mousePoints;
Point points;

void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{

	Mat hh;
	hh = *(Mat*)userdata;
	Point p(x, y);
	switch (EVENT)
	{
	case EVENT_LBUTTONDOWN:
	{
		points.x = x;
		points.y = y;
		mousePoints.push_back(points);
		circle(hh, points, 4, cvScalar(255, 255, 255), -1);
		imshow("mouseCallback", hh);
	}
	break;
	}

}

int selectPolygon(cv::Mat srcMat, cv::Mat &dstMat)
{

	vector<vector<Point>> contours;
	cv::Mat selectMat;

	cv::Mat m = cv::Mat::zeros(srcMat.size(), CV_32F);

	m = 1;

	if (!srcMat.empty()) {
		srcMat.copyTo(selectMat);
		srcMat.copyTo(dstMat);
	}
	else {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	namedWindow("mouseCallback");
	imshow("mouseCallback", selectMat);
	setMouseCallback("mouseCallback", on_mouse, &selectMat);
	waitKey(0);
	destroyAllWindows();
	//计算roi
	contours.push_back(mousePoints);
	if (contours[0].size() < 3) {
		std::cout << "failed to read image!:" << std::endl;
		return -1;
	}

	drawContours(m, contours, 0, Scalar(0), -1);

	m.copyTo(dstMat);

	return 0;
}


int main()
{
	cv::Mat dst1;
	cv::Mat dst2;
	cv::Mat src1 = imread("1.png", 0);
	cv::Mat src2 = imread("2.png", 0);

	int m1 = getOptimalDFTSize(src1.rows); //2,3,5的倍数有更高效率的傅里叶变换
	int n1 = getOptimalDFTSize(src1.cols);
	int m2 = getOptimalDFTSize(src2.rows); //2,3,5的倍数有更高效率的傅里叶变换
	int n2 = getOptimalDFTSize(src2.cols);
	Mat padded1;
	Mat padded2;
	//把灰度图像放在左上角,在右边和下边扩展图像,扩展部分填充为0;
	copyMakeBorder(src1, padded1, 0, m1 - src1.rows, 0, n1 - src1.cols, BORDER_CONSTANT, Scalar::all(0));
	copyMakeBorder(src2, padded2, 0, m2 - src2.rows, 0, n2 - src2.cols, BORDER_CONSTANT, Scalar::all(0));
	//planes[0]为dft变换的实部，planes[1]为虚部，ph为相位， plane_true=mag为幅值
	Mat planes1[] = { Mat_<float>(padded1), Mat::zeros(padded1.size(), CV_32F) };
	Mat planes2[] = { Mat_<float>(padded2), Mat::zeros(padded2.size(), CV_32F) };
	Mat planes_true1 = Mat_<float>(padded1);
	Mat planes_true2 = Mat_<float>(padded2);
	Mat ph1 = Mat_<float>(padded1);
	Mat ph2 = Mat_<float>(padded2);
	Mat complexImg1;
	Mat complexImg2;

	//多通道complexImg既有实部又有虚部
	merge(planes1, 2, complexImg1);
	merge(planes2, 2, complexImg2);
	//对上边合成的mat进行傅里叶变换,***支持原地操作***,傅里叶变换结果为复数.通道1存的是实部,通道二存的是虚部
	dft(complexImg1, complexImg1);
	dft(complexImg2, complexImg2);
	//把变换后的结果分割到两个mat,一个实部,一个虚部,方便后续操作
	split(complexImg1, planes1);
	split(complexImg2, planes2);

	//---------------此部分目的为更好地显示幅值---后续恢复原图时反着再处理一遍-------------------------
	magnitude(planes1[0], planes1[1], planes_true1);//幅度谱mag
	magnitude(planes2[0], planes2[1], planes_true2);
	phase(planes1[0], planes1[1], ph1);//相位谱ph
	phase(planes2[0], planes2[1], ph2);
	Mat A1 = planes1[0];
	Mat B1 = planes1[1];
	Mat mag1 = planes_true1;
	Mat A2 = planes2[0];
	Mat B2 = planes2[1];
	Mat mag2 = planes_true2;

	mag1 += Scalar::all(1);//对幅值加1
	mag2 += Scalar::all(1);//对幅值加1
	log(mag1, mag1);//计算出的幅值一般很大，达到10^4,通常没有办法在图像中显示出来，需要对其进行log求解。
	log(mag2, mag2);
				  //取矩阵中的最大值，便于后续还原时去归一化
	double maxVal;
	minMaxLoc(mag1, 0, &maxVal, 0, 0);
	minMaxLoc(mag2, 0, &maxVal, 0, 0);

	//修剪频谱,如果图像的行或者列是奇数的话,那其频谱是不对称的,因此要修剪
	mag1 = mag1(Rect(0, 0, mag1.cols & -2, mag1.rows & -2));
	ph1 = ph1(Rect(0, 0, mag1.cols & -2, mag1.rows & -2));
	Mat _magI1 = mag1.clone();
	//将幅度归一化到可显示范围。
	normalize(_magI1, _magI1, 0, 1, CV_MINMAX);
	//imshow("before rearrange", _magI);
	mag2 = mag2(Rect(0, 0, mag2.cols & -2, mag2.rows & -2));
	ph2 = ph2(Rect(0, 0, mag2.cols & -2, mag2.rows & -2));
	Mat _magI2 = mag2.clone();
	//将幅度归一化到可显示范围。
	normalize(_magI2, _magI2, 0, 1, CV_MINMAX);

	//显示规则频谱图
	int cx1 = mag1.cols / 2;
	int cy1 = mag1.rows / 2;
	int cx2 = mag2.cols / 2;
	int cy2 = mag2.rows / 2;

	//这里是以中心为标准，把mag图像分成四部分
	Mat tmp1;
	Mat q01(mag1, Rect(0, 0, cx1, cy1));
	Mat q11(mag1, Rect(cx1, 0, cx1, cy1));
	Mat q21(mag1, Rect(0, cy1, cx1, cy1));
	Mat q31(mag1, Rect(cx1, cy1, cx1, cy1));
	Mat tmp2;
	Mat q02(mag2, Rect(0, 0, cx2, cy2));
	Mat q12(mag2, Rect(cx2, 0, cx2, cy2));
	Mat q22(mag2, Rect(0, cy2, cx2, cy2));
	Mat q32(mag2, Rect(cx2, cy2, cx2, cy2));

	q01.copyTo(tmp1);
	q31.copyTo(q01);
	tmp1.copyTo(q31);
	q11.copyTo(tmp1);
	q21.copyTo(q11);
	tmp1.copyTo(q21);

	q02.copyTo(tmp2);
	q32.copyTo(q02);
	tmp2.copyTo(q32);
	q12.copyTo(tmp2);
	q22.copyTo(q12);
	tmp2.copyTo(q22);

	normalize(mag1, mag1, 0, 1, CV_MINMAX);
	normalize(mag2, mag2, 0, 1, CV_MINMAX);
	//imshow("原图灰度图", src);
	//imshow("频谱幅度", mag);
	mag1 = mag1 * 255;
	imwrite("原频谱.jpg", mag1);
	mag2 = mag2 * 255;
	imwrite("原频谱.jpg", mag2);
	/*--------------------------------------------------*/

	mag1 = mag1 / 255;
	mag2 = mag2 / 255;
	cv::Mat mask1;
	cv::Mat mask2;
	Mat proceMag1; 
	Mat proceMag2;


	selectPolygon(mag1, mask1);
	selectPolygon(mag2, mask2);

	mag1 = mag1.mul(mask1);
	mag2 = mag2.mul(mask2);

	proceMag1 = mag1 * 255;
	imwrite("处理后频谱1.jpg", proceMag1);
	proceMag2 = mag2 * 255;
	imwrite("处理后频谱2.jpg", proceMag2);

	//前述步骤反着来一遍，目的是为了逆变换回原图
	Mat q001(mag1, Rect(0, 0, cx1, cy1));
	Mat q101(mag1, Rect(cx1, 0, cx1, cy1));
	Mat q201(mag1, Rect(0, cy1, cx1, cy1));
	Mat q301(mag1, Rect(cx1, cy1, cx1, cy1));

	Mat q002(mag2, Rect(0, 0, cx2, cy2));
	Mat q102(mag2, Rect(cx2, 0, cx2, cy2));
	Mat q202(mag2, Rect(0, cy2, cx2, cy2));
	Mat q302(mag2, Rect(cx2, cy2, cx2, cy2));

	//交换象限
	q001.copyTo(tmp1);
	q301.copyTo(q001);
	tmp1.copyTo(q301);
	q101.copyTo(tmp1);
	q201.copyTo(q101);
	tmp1.copyTo(q201);

	q002.copyTo(tmp2);
	q302.copyTo(q002);
	tmp2.copyTo(q302);
	q102.copyTo(tmp2);
	q202.copyTo(q102);
	tmp2.copyTo(q202);

	mag1 = mag1 * maxVal;//将归一化的矩阵还原 
	exp(mag1, mag1);//对应于前述去对数
	mag1 = mag1 - Scalar::all(1);//对应前述+1
	polarToCart(mag1, ph1, planes1[0], planes1[1]);//由幅度谱mag和相位谱ph恢复实部planes[0]和虚部planes[1]
	merge(planes1, 2, complexImg1);//将实部虚部合并

	mag2 = mag2 * maxVal;//将归一化的矩阵还原 
	exp(mag2, mag2);//对应于前述去对数
	mag2 = mag2 - Scalar::all(1);//对应前述+1
	polarToCart(mag2, ph2, planes2[0], planes2[1]);//由幅度谱mag和相位谱ph恢复实部planes[0]和虚部planes[1]
	merge(planes2, 2, complexImg2);//将实部虚部合并


	//-----------------------傅里叶的逆变换-----------------------------------
	Mat ifft1(Size(src1.cols, src1.rows), CV_8UC1);
	//傅里叶逆变换
	idft(complexImg1, ifft1, DFT_REAL_OUTPUT);
	normalize(ifft1, ifft1, 0, 1, CV_MINMAX);

	Mat ifft2(Size(src2.cols, src2.rows), CV_8UC1);
	//傅里叶逆变换
	idft(complexImg2, ifft2, DFT_REAL_OUTPUT);
	normalize(ifft2, ifft2, 0, 1, CV_MINMAX);

	Rect rect1(0, 0, src1.cols, src1.rows);
	dst1 = ifft1(rect1);
	dst1 = dst1 * 255;
	Rect rect2(0, 0, src2.cols, src2.rows);
	dst2 = ifft2(rect2);
	dst2 = dst2 * 255;

	cv::Mat dspMat1;
	cv::Mat dspMat2;
	cv::Mat image;
	dst1.convertTo(dspMat1, CV_8UC1);
	dst2.convertTo(dspMat2, CV_8UC1);
	imshow("dst1", dspMat1);
	imshow("dst2", dspMat2);
	imshow("sr1c", src1);
	imshow("sr1c", src2);
	addWeighted(src1, 0.5, src2, 0.5, 0, image);//0.5+0.5=1,0.3+0.7=1
	imshow("image", image);
	waitKey(0);

	return 0;

}
