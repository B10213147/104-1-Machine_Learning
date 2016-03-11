/* Includes ------------------------------------------------------------------*/
#include "opencv2/core/mat.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <fstream> 	//fstream, ofstream
#include <cstdlib>	//atof
#include <cstring>	//strncpy, strtok

/* Private macro -------------------------------------------------------------*/
#define Times 42
#define Size 50
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
Mat W01_last(Neurons+1,input_num + 1,CV_32F);
Mat W01(Neurons+1,input_num + 1,CV_32F);
Mat W01_next(Neurons+1,input_num + 1,CV_32F);
Mat V1(Neurons + 1,1,CV_32F);
Mat Xout1_last(Neurons + 1,1,CV_32F);
Mat Xout1(Neurons + 1,1,CV_32F);
Mat W12_last(Neurons + 1,1,CV_32F);
Mat W12(Neurons + 1,1,CV_32F);
Mat W12_next(Neurons + 1,1,CV_32F);
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
void modify_Buffer(float A[input_num+1][Neurons+1]){
	char temp[Size];
	char *delim = "\t";
	char *pch;
	for(int i = 0; i < Neurons; i++){
		//using strncpy convert string to char
		strncpy(temp, Buffer[i].c_str(),sizeof(temp));
		temp[sizeof(temp) - 1]= 0;

		//using strtok to cut the "/t" down from string
		pch = strtok(temp, delim);
		A[0][i] = atof(pch);
		for(int j = 1; j < input_num + 1; j++){
			pch = strtok(NULL, delim);
			A[j][i] = atof(pch);
		}
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
	    system("exit");
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
    float u0 = 0.0002;
    float k0 = 101;
    float c = 0.02;
    float temp = 0;

    temp = u0*(1 + (c/u0)*(k/k0))/
        (1 + (c/u0)*(k/k0) + k0*pow(k/k0, 2));
    return temp;
}
void compute_V1(int k){
    for(int i=1; i<Neurons+1; i++){
        V1.row(i) = Xin.col(k) * W01.row(i).col(1) +
                    W01.row(i).col(0);
    }
}
void compute_Xout1(int k){
    Xout1_last = Xout1;
    for(int i=1; i<Neurons+1; i++){
        Xout1.row(i) = activ_F(V1.at<float>(i));
    }
}
void compute_V2(int k){
    for(int i=0; i<Neurons+1; i++){
        V2.row(0) += Xout1.row(i) * W12.row(i);
    }
}
void compute_Y(int k){
    Y.col(k) = activ_F(V2.at<float>(0));
}
void compute_Error(int k){
    float e = Desire.at<float>(0, k) - Y.at<float>(0, k);
    E.col(k) = 0.5 * pow(e, 2);
}
void training_W01(int k, float e_sum){
    static float delta_last;
    float delta = e_sum * dactiv_F(V1.at<float>(k));

    if(k>0){
        for(int i=0; i<Neurons+1; i++){
            W01_next.row(i).col(1) = W01.row(i).col(1)+
            learning_Rate(k)*(delta*Xin.col(k)+
                    0.25*delta_last*Xin.col(k-1));
            W01_next.row(i).col(0) = W01.row(i).col(0)+
            learning_Rate(k)*(delta*1+0.25*delta_last*1);
        }
    }
    else{
        for(int i=0; i<Neurons+1; i++){
            W01_next.row(i).col(1) = W01.row(i).col(1)+
            learning_Rate(k)*delta*Xin.col(k);
            W01_next.row(i).col(0) = W01.row(i).col(0)+
            learning_Rate(k)*delta*1;
        }
    }
/*
    if(k>0){
        for(int i=0; i<Neurons+1; i++){
            W01_next.row(i).col(1) = W01.row(i).col(1)+
            learning_Rate(k)*delta*Xin.col(k);
            W01_next.row(i).col(0) = W01.row(i).col(0)+
            learning_Rate(k)*delta*1;
        }
    }
    else{
        for(int i=0; i<Neurons+1; i++){
            W01_next.row(i).col(1) = W01.row(i).col(1)+
            learning_Rate(k)*delta*Xin.col(k);
            W01_next.row(i).col(0) = W01.row(i).col(0)+
            learning_Rate(k)*delta*1;
        }
    }
*/
    delta_last = delta;
}
float training_W12(int k){
    static float delta_last;

    float e = Desire.at<float>(0,k) - Y.at<float>(0,k);

    float delta = e * dactiv_F(V2.at<float>(0,0));

    float e_sum = 0;
    for(int i=1; i<Neurons+1; i++){
        e_sum += delta*W12.at<float>(i,0);
    }
    //cout<<"check1"<<endl;

    if(k>0){
        for(int i=0; i<Neurons+1; i++){
            W12_next.row(i) = W12.row(i)+
            learning_Rate(k)*(delta*Xout1.row(i)+
                    0.25*delta_last*Xout1_last.row(i));
        }
    }
    else{
        for(int i=0; i<Neurons+1; i++){
            W12_next.row(i) = W12.row(i)+
            learning_Rate(k)*delta*Xout1.row(i);
        }
    }
/*
    if(k>0){
        for(int i=0; i<Neurons+1; i++){
            W12_next.row(i) = W12.row(i)+
            learning_Rate(k)*delta*Xout1.row(i);
        }
    }
    else{
        for(int i=0; i<Neurons+1; i++){
            W12_next.row(i) = W12.row(i)+
            learning_Rate(k)*delta*Xout1.row(i);
        }
    }
*/
    delta_last = delta;
    return e_sum;
}

/* Private define ------------------------------------------------------------*/
float Temp0[input_num+1][Neurons+1];
float Temp1[Neurons + 1];
float Temp2[Times];

/* Main function -------------------------------------------------------------*/
int main()
{
    int k=0;

    cout<<"Enter hidden layer's weight file:";
    open_file();
    modify_Buffer(Temp0);
    Mat W = Mat(input_num+1,Neurons+1,CV_32F,Temp0).clone();
    W01 = W.t();
    cout<<"Enter output layer's weight file:";
    open_file();
    modify_Buffer(Temp1);
    W = Mat(1,Neurons+1,CV_32F,Temp1).clone();

    W12 = W.t();
    cout<<"Enter desire file:";
    open_file();
    modify_Buffer(Temp2);
    Desire = Mat(output_num,Times,CV_32F,Temp2).clone();

    int i=0;
    for(float j=0; j<=4; j+=0.2){
        Xin.col(i) = j;
        Xin.col(i+21) = j;
        i++;
    }
    Xout1.row(0) = 1;
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

        W01_last = W01; W01 = W01_next;
        W12_last = W12; W12 = W12_next;
    }
    cout<<"k="<<k<<endl;
    cout<<"training end"<<endl;

    ofstream output("W01_training.txt", ios::out);
    for(int i=0; i<Neurons+1; i++){
		for(int j=0; j<input_num+1; j++) output<<W01.at<float>(i,j)<<"\t";
		output<<endl;
	}
	output.close();
	delete output;

	ofstream output1("W01_last_training.txt", ios::out);
    for(int i=0; i<Neurons+1; i++){
		for(int j=0; j<input_num+1; j++) output1<<W01_last.at<float>(i,j)<<"\t";
		output1<<endl;
	}
	output1.close();

	ofstream output2("W12_last_training.txt", ios::out);
    for(int i=0; i<Neurons+1; i++){
		for(int j=0; j<1; j++) output2<<W12_last.at<float>(i,j)<<"\t";
		output2<<endl;
	}
	output2.close();

	ofstream output3("W12_training.txt", ios::out);
    for(int i=0; i<Neurons+1; i++){
		for(int j=0; j<1; j++) output3<<W12.at<float>(i,j)<<"\t";
		output3<<endl;
	}
	output3.close();

	ofstream output4("Error.txt", ios::out);
    for(int i=0; i<Times; i++){
		for(int j=0; j<1; j++) output4<<E.t().row(i).col(j)<<"\t";
		output4<<endl;
	}
	output4.close();

	ofstream output5("Y.txt", ios::out);
    for(int i=0; i<Times; i++){
		for(int j=0; j<1; j++) output5<<Y.t().row(i).col(j)<<"\t";
		output5<<endl;
	}
	output5.close();

    system("pause");
    return 0;
}

