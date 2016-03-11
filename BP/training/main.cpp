/* Includes ------------------------------------------------------------------*/
#include "opencv2/core/mat.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <fstream> 	//fstream, ofstream
#include <cstdlib>	//atof
#include <cstring>	//strncpy, strtok

/* Private macro -------------------------------------------------------------*/
#define Times 21
#define Size 60
#define Neurons 50
#define Empty 0
#define input_num 1
#define output_num 1

using namespace cv;
using namespace std;

/* Private define ------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
static string Buffer[Size];
Mat Xin(input_num,Times,CV_32F);
Mat W01(Neurons,Times + 1,CV_32F);
Mat V1(Neurons,1,CV_32F);
Mat Xout1(Neurons,1,CV_32F);
Mat W12(Neurons,Times + 1,CV_32F);
Mat V2(output_num,1,CV_32F);
Mat Y(output_num,Times,CV_32F);
Mat Desire(output_num,Times,CV_32F);
Mat E(output_num,Times,CV_32F);


/* Private functions ---------------------------------------------------------*/
void modify_Buffer(float *A){
	char temp[Size];

	for(int i = 0; i < Neurons; i++){
		/*using strncpy convert string to char*/
		strncpy(temp, Buffer[i].c_str(),sizeof(temp));
		temp[sizeof(temp) - 1]= 0;

		A[i] = atof(temp);
	}
}

void open_file(void) {
	fstream file;
	char fname[Size];
	cin>>fname;
	cout<<endl<<"Opening file:"<<fname<<endl;
	file.open(fname,ios::in);	//file read only
	if(!file){
      	cout<<"File I/O failed!"<<endl;
	    system("pause");
	}

	string Line;
	for(int i=0;i<Size;i++){
		getline(file, Line);
		if(Line[0] == Empty) break;	//break for loop when file doesn't have vaule
		else{
			Buffer[i] = Line;
		}
	}
	cout<<"file closed."<<endl;
	file.close();
}
float activ_F(float x){
    return tanh(x);
}
float dactiv_F(float x){
    return 1 - pow(tanh(x), 2);
}
void compute_V1(int k){
    for(int i=0; i<Neurons; i++){
        V1.row(i) = Xin.col(k) * W01.row(i).col(k);
    }
}
void compute_Xout1(int k){
    for(int i=0; i<Neurons; i++){
        Xout1.row(i) = activ_F(V1.at<float>(i));
    }
}
void compute_V2(int k){
    for(int i=0; i<output_num; i++){
        V2.row(i) += Xout1.row(i) * W12.row(i).col(k);
    }
}
void compute_Y(int k){
    Y.col(k) = activ_F(V2.at<float>(0));
}
void compute_Error(int k){
    float e = Desire.at<float>(k) - Y.at<float>(k);
    E.col(k) = 0.5 * pow(e, 2);
}
void training_W01(int k){

}
void training_W12(int k){

}

/* Private define ------------------------------------------------------------*/
float Temp[Neurons];
float Temp2[Times];

/* Main function -------------------------------------------------------------*/
int main()
{
    int k=0;

    cout<<"Enter hidden layer's weight file:";
    open_file();
    modify_Buffer(Temp);
    Mat W = Mat(Times,Neurons,CV_32F,Temp).clone();
    //Mat W01(Neurons,Times,CV_32F,Scalar(0));
    W01 = W.t();
    cout<<"Enter output layer's weight file:";
    open_file();
    modify_Buffer(Temp);
    W = Mat(Times,Neurons,CV_32F,Temp).clone();
    //Mat W12(Neurons,Times,CV_32F,Scalar(0));
    W12 = W.t();
    cout<<"Enter desire file:";
    open_file();
    modify_Buffer(Temp2);
    Desire = Mat(output_num,Times,CV_32F,Temp2).clone();

    for(int i=0; i<Times; i++){
        for(float j=0; j<4; j+=0.2) Xin.col(i) = j;
    }

    while(k < Times){
        compute_V1(k);
        compute_Xout1(k);
        compute_V2(k);
        compute_Y(k);
        compute_Error(k);
        training_W01(k);
        training_W12(k);
        k++;
    }


    return 0;
}

