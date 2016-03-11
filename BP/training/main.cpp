/* Includes ------------------------------------------------------------------*/
#include "opencv2/core/mat.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <fstream> 	//fstream, ofstream
#include <cstdlib>	//atof
#include <cstring>	//strncpy, strtok

/* Private macro -------------------------------------------------------------*/
#define Times 105
#define Size 110
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
Mat Xout1(Neurons,2,CV_32F);
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
float learning_Rate(int k){
    float u0 = 0.01;
    float k0 = 200;
    float c = 0.1;
    float temp = 0;

    temp = u0*(1 + (c/u0)*(k/k0))/
        (1 + (c/u0)*(k/k0) + k0*pow(k/k0, 2));
    return temp;
}
void compute_V1(int k){
    for(int i=0; i<Neurons; i++){
        V1.row(i) = Xin.col(k) * W01.row(i).col(k);
    }
}
void compute_Xout1(int k){
    for(int i=0; i<Neurons; i++){
        Xout1.row(i).col(0) = Xout1.row(i).col(1);
    }
    for(int i=0; i<Neurons; i++){
        Xout1.row(i).col(1) = activ_F(V1.at<float>(i));
    }
}
void compute_V2(int k){
    for(int i=0; i<output_num; i++){
        V2.row(i) += Xout1.row(i).col(1) * W12.row(i).col(k);
    }
}
void compute_Y(int k){
    Y.col(k) = activ_F(V2.at<float>(0));
}
void compute_Error(int k){
    float e = Desire.at<float>(k) - Y.at<float>(k);
    E.col(k) = 0.5 * pow(e, 2);
}
void training_W01(int k, float e_sum){
    static float delta_last;
    float delta = e_sum * dactiv_F(V1.at<float>(k));
    if(k==0){

        for(int i=0; i<Neurons; i++){
        W01.row(i).col(k+1)=W01.row(i).col(k)+
        learning_Rate(k)*delta*Xin.col(k);
        }
    }
    else{
    for(int i=0; i<Neurons; i++){
        W01.row(i).col(k+1)=W01.row(i).col(k)+
        learning_Rate(k)*(delta*Xin.col(k)+
            0.25*delta_last*Xin.col(k-1));
    }
    }

    delta_last = delta;
}
float training_W12(int k){
    static float delta_last;

    float e = Desire.at<float>(k) - Y.at<float>(k);

    float delta = e * dactiv_F(V2.at<float>(0,0));

    float e_sum = 0;
    for(int i=0; i<Neurons; i++){
        e_sum += delta * W12.at<float>(i,k);
    }
    if(k==0){
        for(int i=0; i<Neurons; i++){
        W12.row(i).col(k+1)=W12.row(i).col(k)+
        learning_Rate(k)*delta*Xout1.row(i).col(1);
        }
    }
    for(int i=0; i<Neurons; i++){
        W12.row(i).col(k+1)=W12.row(i).col(k)+
        learning_Rate(k)*(delta*Xout1.row(i).col(1)+
                0.25*delta_last*Xout1.row(i).col(0));
    }
    delta_last = delta;
    return e_sum;
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
    Mat W = Mat(Times + 1,Neurons,CV_32F,Temp).clone();
    //Mat W01(Neurons,Times,CV_32F,Scalar(0));
    W01 = W.t();
    cout<<"Enter output layer's weight file:";
    open_file();
    modify_Buffer(Temp);
    W = Mat(Times + 1,Neurons,CV_32F,Temp).clone();
    //Mat W12(Neurons,Times,CV_32F,Scalar(0));
    W12 = W.t();
    cout<<"Enter desire file:";
    open_file();
    modify_Buffer(Temp2);
    Desire = Mat(output_num,Times,CV_32F,Temp2).clone();

    int i=0;
    for(float j=0; j<=4; j+=0.2){
        Xin.col(i) = j;
        Xin.col(i+21) = j;
        Xin.col(i+42) = j;
        Xin.col(i+63) = j;
        Xin.col(i+84) = j;
        i++;
    }
    //cout<<"check0"<<endl;
    while(k < Times){
        compute_V1(k);
        //cout<<"check1"<<endl;
        compute_Xout1(k);
        //cout<<"check2"<<endl;
        compute_V2(k);
        //cout<<"check3"<<endl;
        compute_Y(k);
        //cout<<"check4"<<endl;
        compute_Error(k);
        //cout<<"check5"<<endl;
        float e_sum = training_W12(k);
        //cout<<"check1"<<endl;
        training_W01(k, e_sum);
        //cout<<"check2"<<endl;
        k++;
        cout<<"k="<<k<<endl;
    }
    cout<<"training end"<<endl;
    cout<<"Error="<<endl<<E<<endl;

    system("pause");
    return 0;
}

