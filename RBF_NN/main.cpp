/* Includes ------------------------------------------------------------------*/
#include "opencv2/core/mat.hpp"
#include "opencv2/core/utility.hpp"
#include <iostream>
#include <fstream> 	//fstream, ofstream
#include <cstdlib>	//atof
#include <cstring>	//strncpy, strtok
#include <ctime>

/* Private macro -------------------------------------------------------------*/
#define Times 21
#define Neurons 21
#define center_num 21
#define input_num 1
#define output_num 1

using namespace cv;
using namespace std;

/* Private define ------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
Mat Xin(input_num,Times,CV_32F);
Mat RBF_out(Neurons + 1, input_num, CV_32F);
Mat Center(Neurons + 1, 1, CV_32F);
Mat Center_next(Neurons + 1, 1, CV_32F);
Mat Spread(Neurons + 1, 1, CV_32F);
Mat Spread_next(Neurons + 1, 1, CV_32F);
Mat W1(Neurons + 1, 1, CV_32F);
Mat W1_next(Neurons + 1, 1, CV_32F);
Mat Y(output_num,Times,CV_32F);
Mat Desire(output_num,Times,CV_32F);
Mat E(output_num,Times,CV_32F);


/* Private functions ---------------------------------------------------------*/
void initial_Spread(void){
    float max = 0;
    float min = 0;

    for(int i=0; i<Times; i++){
        if(Xin.at<float>(0,i) > max){
            max = Xin.at<float>(0,i);
        }
        if(Xin.at<float>(0,i) < min){
            min = Xin.at<float>(0,i);
        }
    }
    float dmax = max - min;
    Spread = dmax / sqrt(Neurons);
}
float error_goal(void){
    return pow(10, -3);
}
float learning_Rate(int k){
    const float u0 = 0.20;
    const float k0 = 100;
    const float c = 18;

    return u0*(1 + (c/u0)*(k/k0))/
        (1 + (c/u0)*(k/k0) + k0*pow(k/k0, 2));
}
float RBF(float x, float c, float s){
    return exp(-pow(x-c, 2) / pow(s, 2));
}
void build_RBF_out(int k){
    for(int i=1; i<Neurons + 1; i++){
        RBF_out.row(i) = RBF(Xin.at<float>(0,k), Center.at<float>(i,0), Spread.at<float>(i,0));
    }
}
void compute_Y(int k){
    Y.col(k) = RBF_out.t() * W1;
}
float compute_Error(int k){
    float e = Desire.at<float>(0, k) - Y.at<float>(0, k);
    E.col(k) = 0.5 * pow(e, 2);
    return e;
}
void training_W1(int k, float u, float e){
    for(int i=0; i<Neurons+1; i++){
        W1_next.row(i) = W1.row(i) + u * e * RBF_out.row(i);
    }
}
void training_Center(int k, float u, float e){
    for(int i=1; i<Neurons+1; i++){
        Center_next.row(i) = Center.row(i) +
            u * e * W1.row(i) *RBF_out.row(i) *
            (Xin.col(k) - Center.row(i)) / pow(Spread.at<float>(i, 0), 2);
    }
}
void training_Spread(int k, float u, float e){
    for(int i=1; i<Neurons+1; i++){
        Spread_next.row(i) = Spread.row(i) +
            u * e * W1.row(i) * RBF_out.row(i) *
            pow(Xin.at<float>(0, k) - Center.at<float>(i, 0), 2)/
            pow(Spread.at<float>(i, 0), 3);
    }
}

/* Private define ------------------------------------------------------------*/
/* Main function -------------------------------------------------------------*/
int main()
{
    srand(time(NULL));
    for(int i=0; i<Neurons+1; i++){
        W1.row(i) = 0.1 * rand() / RAND_MAX - 0.05;
    }

    int k_total=0;
    static bool flag=1;
    int i=0;
    for(float x=0; x<4.1; x+=0.2){
        Xin.col(i) = x;
        Desire.col(i) = exp(-x) * sin(3 * x);
        Center.row(i+1) = x;
        i++;
    }
    initial_Spread();
    RBF_out.row(0) = 1;

    do{
        for(int k=0; k<Times; k++){
            build_RBF_out(k);

            compute_Y(k);

            float e = compute_Error(k);

            float u = learning_Rate(k_total);

            training_W1(k, u, e);
            training_Center(k, u, e);
            training_Spread(k, u, e);

            W1 = W1_next;
            Center = Center_next;
            Spread = Spread_next;
        }

        float e_max = 0;
        int n = 0;
        for(int i=0; i<Times; i++){
            float temp = E.at<float>(0,i);
            if(temp > error_goal()){
                if(temp > e_max){
                    e_max = temp;
                    flag =1;
                }
                else flag = 1;
            }
            else{
                n++;
                if(n==Times - 1){
                    flag = 0;
                    break;
                }
            }
        }
        if(k_total % 25 == 0){
            system("CLS");
            cout<<"Remain:"<<(e_max-error_goal())/error_goal()<<endl;
        }
        k_total++;
    }while(flag==1 && k_total<5000);
    cout<<"k_total="<<k_total<<endl;
    cout<<"training end"<<endl;

	ofstream output1("Center_training.txt", ios::out);
    for(int i=0; i<Neurons+1; i++){
		for(int j=0; j<1; j++) output1<<Center.at<float>(i,j)<<endl;
	}
	output1.close();

	ofstream output2("Spread_training.txt", ios::out);
    for(int i=0; i<Neurons+1; i++){
		for(int j=0; j<1; j++) output2<<Spread.at<float>(i,j)<<endl;
	}
	output2.close();

	ofstream output3("W1_training.txt", ios::out);
    for(int i=0; i<Neurons+1; i++){
		for(int j=0; j<1; j++) output3<<W1.at<float>(i,j)<<endl;
	}
	output3.close();

	ofstream output4("Error.txt", ios::out);
    for(int i=0; i<Times; i++){
		for(int j=0; j<1; j++) output4<<E.at<float>(j,i)<<endl;
	}
	output4.close();

	ofstream output5("Y.txt", ios::out);
    for(int i=0; i<Times; i++){
		for(int j=0; j<1; j++) output5<<Y.at<float>(j,i)<<endl;
	}
	output5.close();

    system("pause");
    return 0;
}

