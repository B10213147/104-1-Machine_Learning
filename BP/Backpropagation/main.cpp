/* Includes ------------------------------------------------------------------*/
#include "cv.h"
#include "highgui.h"
#include "GraphUtils.h"
#include <iostream>
#include <fstream> 	//fstream, ofstream
#include <cstdlib>	//atof
#include <cstring>	//strncpy, strtok
#include <ctime>

/* Private macro -------------------------------------------------------------*/
#define training_Times 21
#define testing_Times 401
#define rand_between 2.0    //+2.0 ~ -2.0
#define Neurons 50
#define input_num 1
#define output_num 1

using namespace cv;
using namespace std;

/* Private define ------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
Mat Xin(input_num, testing_Times, CV_32F);
Mat W01_last(Neurons+1, input_num+1, CV_32F);
Mat W01(Neurons+1, input_num+1, CV_32F);
Mat W01_next(Neurons+1, input_num+1, CV_32F);
Mat V1(Neurons+1, 1, CV_32F);
Mat Xout1_last(Neurons+1, 1, CV_32F);
Mat Xout1(Neurons+1, 1, CV_32F);
Mat W12_last(Neurons+1, 1, CV_32F);
Mat W12(Neurons+1, 1, CV_32F);
Mat W12_next(Neurons+1, 1, CV_32F);
Mat e_sum(Neurons+1, 1, CV_32F);
Mat V2(output_num, 1, CV_32F);
Mat Y(output_num, testing_Times, CV_32F);
Mat Desire(output_num, testing_Times, CV_32F);
Mat E(output_num, testing_Times, CV_32F);

/* Private functions ---------------------------------------------------------*/
float activ_F(float x){
    return tanh(x);
}
float dactiv_F(float x){
    return 1 - pow(tanh(x), 2);
}
float target_function(float x){
    return exp(-x) * sin(3*x) ;
}
float error_goal(void){
    return pow(10, -3);
}
float learning_Rate(int k){
    const float u0 = 0.08;
    const float k0 = 350;
    const float c = 7;

    return u0*(1 + (c/u0) * (k/k0))/
        (1 + (c/u0) * (k/k0) + k0 * pow(k/k0, 2));
}
void compute_V1(int k){
    for(int i=1; i < Neurons+1; i++){
        V1.row(i) = Xin.col(k) * W01.row(i).col(1) +
                    W01.row(i).col(0);
    }
}
void compute_Xout1(void){
    Xout1_last = Xout1;
    for(int i=1; i < Neurons+1; i++){
        Xout1.row(i) = activ_F(V1.at<float>(i));
    }
}
void compute_V2(void){
    V2.row(0) = Xout1.t() * W12;
}
void compute_Y(int k){
    Y.col(k) = V2.at<float>(0,0);
}
float compute_Error(int k){
    float e = Desire.at<float>(0, k) - Y.at<float>(0, k);
    E.col(k) = 0.5 * pow(e, 2);
    return e;
}
void training_W01(int k, float u){
    static float delta_last[Neurons+1];

    if(k>0){
        for(int i=1; i < Neurons+1; i++){
            float delta = e_sum.at<float>(i) * dactiv_F(V1.at<float>(i));
            W01_next.row(i).col(1) = W01.row(i).col(1)+
            u * (delta * Xin.col(k) + 0.005*delta_last[i] * Xin.col(k-1));
            W01_next.row(i).col(0) = W01.row(i).col(0)+
            u * (delta*1 + 0.005*delta_last[i]*1);
            delta_last[i] = delta;
        }
    }
    else{
        for(int i=1; i < Neurons+1; i++){
            float delta = e_sum.at<float>(i) * dactiv_F(V1.at<float>(i));
            W01_next.row(i).col(1) = W01.row(i).col(1)+
            u * delta * Xin.col(k);
            W01_next.row(i).col(0) = W01.row(i).col(0) + u * delta*1;
            delta_last[i] = delta;
        }
    }
}
void training_W12(int k, float u, float e){
    static float delta_last;

    float delta = e * dactiv_F(V2.at<float>(0, 0));

    for(int i=1; i < Neurons+1; i++){
        e_sum.row(i) = delta * W12.at<float>(i, 0);
    }

    if(k>0){
        for(int i=0; i < Neurons+1; i++){
            W12_next.row(i) = W12.row(i) + u * (delta * Xout1.row(i)+
                    0.005*delta_last * Xout1_last.row(i));
        }
    }
    else{
        for(int i=0; i < Neurons+1; i++){
            W12_next.row(i) = W12.row(i) + u * delta * Xout1.row(i);
        }
    }
    delta_last = delta;
}

/* Private define ------------------------------------------------------------*/
static bool training_flag;
static bool testing_flag;

/* Main function -------------------------------------------------------------*/
int main()
{
    cout<<"=====This is Back Propagation====="<<endl;
    while(1){
        string temp;
        cout<<"Training?(y/n)";
        cin>>temp;
        if(temp == "y" || temp == "Y"){
            training_flag = 1;
            break;
        }
        else if(temp == "n" || temp == "N"){
            training_flag = 0;
            break;
        }
        else{
            cout<<"Please enter correct answer"<<endl;
        }
    }

    if(training_flag == 1){
        srand(time(NULL));
        for(int i=1; i < Neurons+1; i++){
            W01.row(i).col(0) = 2*rand_between * rand() / RAND_MAX - rand_between;
            W01.row(i).col(1) = 2*rand_between * rand() / RAND_MAX - rand_between;
            W12.row(i) = 2*rand_between * rand() / RAND_MAX - rand_between;
        }
        W12.row(0) = 2*rand_between * rand() / RAND_MAX - rand_between;

        int k_total=0;
        bool flag=1;
        int i=0;
        for(float x=0; x<4.1; x+=0.2){
            Xin.col(i) = x;
            Desire.col(i) = target_function(x);
            i++;
        }
        Xout1.row(0) = 1;

        do{
            for(int k=0; k < training_Times; k++){
                compute_V1(k);

                compute_Xout1();

                compute_V2();

                compute_Y(k);

                float e = compute_Error(k);

                float u = learning_Rate(k_total);
                training_W12(k, u, e);
                training_W01(k, u);

                W01_last = W01; W01 = W01_next;
                W12_last = W12; W12 = W12_next;
            }

            float e_max = 0;
            int n = 0;
            for(int i=0; i < training_Times; i++){
                float temp = E.at<float>(0,i);
                if(temp > error_goal()){
                    if(temp > e_max){
                        e_max = temp;
                        flag=1;
                    }
                    else flag=1;
                }
                else{
                    n++;
                    if(n == training_Times-1){
                        flag=0;
                        break;
                    }
                }
            }
            if(k_total%25 == 0){
                system("CLS");
                cout<<"Remain:"<<(e_max - error_goal()) / error_goal()<<endl;
            }
            k_total++;
        }while(flag==1 && k_total < pow(10, 4));
        cout<<"k_total="<<k_total<<endl;
        cout<<"training end"<<endl;

        ofstream output("training_W01.txt", ios::out);
        for(int i=0; i < Neurons+1; i++){
            for(int j=0; j < input_num+1; j++) output<<W01.at<float>(i, j)<<"\t";
            output<<endl;
        }
        output.close();

        ofstream output3("training_W12.txt", ios::out);
        for(int i=0; i < Neurons+1; i++){
            for(int j=0; j<1; j++) output3<<W12.at<float>(i, j);
            output3<<endl;
        }
        output3.close();

        ofstream output4("training_Error.txt", ios::out);
        for(int i=0; i < training_Times; i++){
            for(int j=0; j<1; j++) output4<<E.at<float>(j, i);
            output4<<endl;
        }
        output4.close();

        ofstream output5("training_Y.txt", ios::out);
        for(int i=0; i < training_Times; i++){
            for(int j=0; j<1; j++) output5<<Y.at<float>(j, i);
            output5<<endl;
        }
        output5.close();

        float D[training_Times];
        float out[training_Times];
        for(int i=0; i < training_Times; i++){
            D[i] = Desire.at<float>(i);
            out[i] = Y.at<float>(i);
        }
        setGraphColor(0);
        IplImage *graphImg = drawFloatGraph(D, training_Times, NULL,
                            -1, 1, 480, 720,
                            "Blue is desire, Green is testing output");
        drawFloatGraph(out, training_Times, graphImg, -1, 1, 480, 720);
        showImage(graphImg, 0, "Desire and Testing compare");
        cvReleaseImage(&graphImg);
    }

    while(1){
        string temp;
        cout<<"Testing?(y/n)";
        cin>>temp;
        if(temp == "y" || temp == "Y"){
            testing_flag=1;
            break;
        }
        else if(temp == "n" || temp == "N"){
            testing_flag=0;
            break;
        }
        else{
            cout<<"Please enter correct answer"<<endl;
        }
    }

    if(testing_flag == 1){
        bool check=1;
        if(training_flag == 0){
            ifstream input1("training_W01.txt");
            if(input1.is_open()){
                string temp[Neurons+1][3];
                for(int i=0; i < Neurons+1; i++){
                        input1 >> temp[i][0];
                        W01.row(i).col(0) = atof(temp[i][0].c_str());
                        input1 >> temp[i][2];
                        W01.row(i).col(1) = atof(temp[i][2].c_str());
                }
                cout<<"W01="<<W01<<endl;
                system("pause");
                check=1;
            }
            else{
                cout<<"Did not find training_W01.txt"<<endl;
                check=0;
            }
            input1.close();

            ifstream input2("training_W12.txt");
            if(input2.is_open()){
                string temp[Neurons+1];
                for(int i=0; i < Neurons+1; i++){
                        input2 >> temp[i];
                        W12.row(i) = atof(temp[i].c_str());
                }
                cout<<"W12="<<W12<<endl;
                system("pause");
                check=1;
            }
            else{
                cout<<"Did not find training_Spread.txt"<<endl;
                check=0;
            }
            input2.close();
            Xout1.row(0) = 1;
        }

        if(check == 1){
            int i=0;
            for(float x=0; x<4.001; x+=0.01){
                Xin.col(i) = x;
                Desire.col(i) = target_function(x);
                i++;
            }

            for(int k=0; k < testing_Times; k++){
                compute_V1(k);

                compute_Xout1();

                compute_V2();

                compute_Y(k);

                compute_Error(k);
            }
            cout<<"Testing_times="<<testing_Times<<endl;
            cout<<"Testing end"<<endl;

            ofstream output4("testing_Error.txt", ios::out);
            for(int i=0; i < testing_Times; i++){
                for(int j=0; j<1; j++) output4<<E.at<float>(j, i)<<endl;
            }
            output4.close();

            ofstream output5("testing_Y.txt", ios::out);
            for(int i=0; i < testing_Times; i++){
                for(int j=0; j<1; j++) output5<<Y.at<float>(j, i)<<endl;
            }
            output5.close();

            float D[testing_Times];
            float out[testing_Times];
            for(int i=0; i < testing_Times; i++){
                D[i] = Desire.at<float>(i);
                out[i] = Y.at<float>(i);
            }
            setGraphColor(0);
            IplImage *graphImg = drawFloatGraph(D, testing_Times, NULL,
                                -1, 1, 360, 480,
                                "Blue is desire, Green is testing output");
            drawFloatGraph(out, testing_Times, graphImg, -1, 1, 360, 480);
            cvSaveImage("1.jpg", graphImg);
            showImage(graphImg, 0, "Desire and Testing compare");
            cvReleaseImage(&graphImg);
        }
    }
    cout<<"=====End of Back Propagation====="<<endl;
    system("pause");
    return 0;
}

