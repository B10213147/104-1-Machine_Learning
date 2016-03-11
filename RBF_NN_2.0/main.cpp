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
#define rand_between 0.05    //+0.05 ~ -0.05
#define Neurons 21
#define input_num 1
#define output_num 1

using namespace cv;
using namespace std;

/* Private define ------------------------------------------------------------*/
/* Private variables ---------------------------------------------------------*/
Mat Xin(input_num, testing_Times, CV_32F);
Mat RBF_out(Neurons+1, input_num, CV_32F);
Mat Center(Neurons+1, 1, CV_32F);
Mat Center_next(Neurons+1, 1, CV_32F);
Mat Spread(Neurons+1, 1, CV_32F);
Mat Spread_next(Neurons+1, 1, CV_32F);
Mat W1(Neurons+1, 1, CV_32F);
Mat W1_next(Neurons+1, 1, CV_32F);
Mat Y(output_num, testing_Times, CV_32F);
Mat Desire(output_num, testing_Times, CV_32F);
Mat E(output_num, testing_Times, CV_32F);

/* Private functions ---------------------------------------------------------*/
void initial_Spread(void){
    float max=0;
    float min=0;

    for(int i=0; i < training_Times; i++){
        if(Xin.at<float>(0, i) > max){
            max = Xin.at<float>(0, i);
        }
        if(Xin.at<float>(0, i) < min){
            min = Xin.at<float>(0, i);
        }
    }
    float dmax = max - min;
    Spread = dmax / sqrt(Neurons);
}
float target_function(float x){
    return sin(-3*x);
}
float error_goal(void){
    return 5*pow(10, -4);
}
float learning_Rate(int k){
    const float u0 = 0.20;
    const float k0 = 100;
    const float c = 18;

    return u0 * (1 + (c/u0) * (k/k0))/
        (1 + (c/u0) * (k/k0) + k0 * pow(k/k0, 2));
}
float RBF(float x, float c, float s){
    return exp(-pow(x-c, 2) / pow(s, 2));
}
void build_RBF_out(int k){
    for(int i=1; i < Neurons+1; i++){
        RBF_out.row(i) = RBF(Xin.at<float>(0, k), Center.at<float>(i, 0), Spread.at<float>(i, 0));
    }
}
void compute_Y(int k){
    Y.col(k) = RBF_out.t() * W1;
}
float compute_Error(int k){
    float e = Desire.at<float>(0, k) - Y.at<float>(0, k);
    E.col(k) = 0.5*pow(e, 2);
    return e;
}
void training_W1(int k, float u, float e){
    for(int i=0; i < Neurons+1; i++){
        W1_next.row(i) = W1.row(i) + u * e * RBF_out.row(i);
    }
}
void training_Center(int k, float u, float e){
    for(int i=1; i < Neurons+1; i++){
        Center_next.row(i) = Center.row(i) +
            u * e * W1.row(i) * RBF_out.row(i) *
            (Xin.col(k) - Center.row(i)) / pow(Spread.at<float>(i, 0), 2);
    }
}
void training_Spread(int k, float u, float e){
    for(int i=1; i < Neurons+1; i++){
        Spread_next.row(i) = Spread.row(i) +
            u * e * W1.row(i) * RBF_out.row(i) *
            pow(Xin.at<float>(0, k) - Center.at<float>(i, 0), 2)/
            pow(Spread.at<float>(i, 0), 3);
    }
}

/* Private define ------------------------------------------------------------*/
static bool training_flag;
static bool testing_flag;

/* Main function -------------------------------------------------------------*/
int main()
{
    cout<<"=====This is Radial Basis Function Neural Networks====="<<endl;
    while(1){
        string temp;
        cout<<"Training?(y/n)";
        cin>>temp;
        if(temp == "y" || temp == "Y"){
            training_flag=1;
            break;
        }
        else if(temp == "n" || temp == "N"){
            training_flag=0;
            break;
        }
        else{
            cout<<"Please enter correct answer"<<endl;
        }
    }

    if(training_flag == 1){
        srand(time(NULL));
        for(int i=0; i < Neurons+1; i++){
            W1.row(i) = 2*rand_between * rand() / RAND_MAX - rand_between;
        }

        int k_total=0;
        static bool flag=1;
        int i=0;
        for(float x=0; x<4.1; x+=0.2){
            Xin.col(i) = x;
            Desire.col(i) = target_function(x);
            Center.row(i+1) = x;
            i++;
        }
        initial_Spread();
        RBF_out.row(0) = 1;

        do{
            for(int k=0; k < training_Times; k++){
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

            float e_max=0;
            int n=0;
            for(int i=0; i < training_Times; i++){
                float temp = E.at<float>(0, i);
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
        }while(flag == 1 && k_total < pow(10, 5));
        cout<<"k_total="<<k_total<<endl;
        cout<<"Training end"<<endl;

        ofstream output1("training_Center.txt", ios::out);
        for(int i=0; i < Neurons+1; i++){
            for(int j=0; j<1; j++) output1<<Center.at<float>(i, j)<<endl;
        }
        output1.close();

        ofstream output2("training_Spread.txt", ios::out);
        for(int i=0; i < Neurons+1; i++){
            for(int j=0; j<1; j++) output2<<Spread.at<float>(i, j)<<endl;
        }
        output2.close();

        ofstream output3("training_W1.txt", ios::out);
        for(int i=0; i < Neurons+1; i++){
            for(int j=0; j<1; j++) output3<<W1.at<float>(i, j)<<endl;
        }
        output3.close();

        ofstream output4("training_Error.txt", ios::out);
        for(int i=0; i < training_Times; i++){
            for(int j=0; j<1; j++) output4<<E.at<float>(j, i)<<endl;
        }
        output4.close();

        ofstream output5("training_Y.txt", ios::out);
        for(int i=0; i < training_Times; i++){
            for(int j=0; j<1; j++) output5<<Y.at<float>(j, i)<<endl;
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
                            -1, 1, 360, 480,
                            "Blue is desire, Green is testing output");
        drawFloatGraph(out, training_Times, graphImg, -1, 1, 360, 480);
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
            ifstream input1("training_Center.txt");
            if(input1.is_open()){
                string temp[Neurons+1];
                for(int i=0; i < Neurons+1; i++){
                        input1 >> temp[i];
                        Center.row(i) = atof(temp[i].c_str());
                }
                cout<<"Center="<<Center<<endl;
                system("pause");
                check=1;
            }
            else{
                cout<<"Did not find training_Center.txt"<<endl;
                check=0;
            }
            input1.close();

            ifstream input2("training_Spread.txt");
            if(input2.is_open()){
                string temp[Neurons+1];
                for(int i=0; i < Neurons+1; i++){
                        input2 >> temp[i];
                        Spread.row(i) = atof(temp[i].c_str());
                }
                cout<<"Spread="<<Spread<<endl;
                system("pause");
                check=1;
            }
            else{
                cout<<"Did not find training_Spread.txt"<<endl;
                check=0;
            }
            input2.close();

            ifstream input3("training_W1.txt");
            if(input3.is_open()){
                string temp[Neurons+1];
                for(int i=0; i < Neurons+1; i++){
                        input3 >> temp[i];
                        W1.row(i) = atof(temp[i].c_str());
                }
                cout<<"W1="<<W1<<endl;
                system("pause");
                check=1;
            }
            else{
                cout<<"Did not find training_W1.txt"<<endl;
                check=0;
            }
            input3.close();
            RBF_out.row(0) = 1;
        }

        if(check == 1){
            int i=0;
            for(float x=0; x<4.001; x+=0.01){
                Xin.col(i) = x;
                Desire.col(i) = target_function(x);
                i++;
            }

            for(int k=0; k < testing_Times; k++){
                build_RBF_out(k);

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
            cvSaveImage("2.jpg", graphImg);
            showImage(graphImg, 0, "Desire and Testing compare");
            cvReleaseImage(&graphImg);
        }
    }
    cout<<"=====End of Radial Basis Function Neuron Networks====="<<endl;
    system("pause");
    return 0;
}

