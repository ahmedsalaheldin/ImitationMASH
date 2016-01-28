#include<stdio.h>
#include<iostream>
#include <fstream>
#include <zmq.hpp>
#include <zmq_utils.h>
#include <string>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include<mash-network/client.h>
//#include <Image.h>
//#include <svm.h>
#include <sstream>

using namespace std;
using namespace Mash;
using namespace rapidjson;


/*void writeImage(unsigned char* imgdata, int width, int height)
{	
	TGAImage *img = new TGAImage(width,height);
	Colour c;
	int count=0;

	for (int y=0; y <height ;y++)
	{
		for (int x=0; x <width ;x++)
		{
			c.r = imgdata[count];
			c.g = imgdata[count+1];	
			c.b = imgdata[count+2];
			c.a = 255;
			img->setPixel(c,y,x);
			count+=3;
		}
	}
	img->WriteImage("test.tga");
}*/

void writeDataset(ofstream* dataset,ofstream* target, unsigned char*data, string action, int datasize )
{
	/*for(int i=0;i<14399;i++)
	{
		*dataset << (int)data[i]<<",";
	}
	*dataset << (int)data[14400];
	*dataset << endl;
	*/// COLOR
	float average; 
	for(int i=0;i<(datasize-3);i+=3)
	{
		//average = ((float)data[i]+(float)data[i+1]+(float)data[i+2])/3;
		average = 0.2126*(float)data[i] + 0.7152*(float)data[i+1] + 0.0722*(float)data[i+2];
		*dataset << (int)average<<",";
	}
	average = ((float)data[(datasize-2)]+(float)data[(datasize-1)]+(float)data[datasize])/3;
		*dataset << (int)average;
	*dataset << endl;

	if( action == "TURN_LEFT")
		*target << 0 <<endl;
	else if( action == "TURN_RIGHT")
		*target << 1 <<endl;
	else if( action == "GO_FORWARD")
		*target << 2 <<endl;
	else if( action == "GO_BACKWARD")
		*target << 3 <<endl;
		
}


int main(int argc, char** argv)
{
	const int width = 120;
	const int height = 90;
	const int datasize = width*height*3;
	int leftcounter=0,rightcounter=0,forwardcounter=0;
	int classlimit =50000;
	bool writeflag =0; 
 	int counter=0 , timeOut = 5000 , numFinish=0; //time counter perlevel  time out =500 for flag, much larger for line 
	ofstream dataset("datasetL2activelearning.csv");
	ofstream target("targetL2activelearning.csv");
	ifstream features;
	OutStream* out = new OutStream(0);
	Client* c = new Client(out);
	unsigned char data[datasize];
	//unsigned char data[230400];
	ArgumentsList arg ;
	ArgumentsList actions ;
	ArgumentsList empty;
	//struct svm_node *x =  new svm_node[3];
	//struct svm_model* model;
	//model=svm_load_model("mysvm");

///////////Connect to Prediction Server/////////////
	/*zmq::context_t context (1);
	zmq::socket_t socket (context, ZMQ_REQ);



	socket.connect ("tcp://localhost:5555");
*/
////////////////////////////////////////////////////////

	string* strResponse = new string;
	ArgumentsList* arguments = new ArgumentsList();
	ArgumentsList* reward = new ArgumentsList(0);
	ArgumentsList* suggestedAction = new ArgumentsList();

	c->connect("127.0.0.1",11200);

	c->sendCommand("STATUS",empty);
	c->waitResponse(strResponse, arguments);
	
	c->sendCommand("USE_GLOBAL_SEED",12345); 
	c->waitResponse(strResponse, arguments);	
	
	c->sendCommand("LIST_GOALS",empty);
	do{
		c->waitResponse(strResponse, arguments);
	
	}while(*strResponse=="GOAL");

	//SELECT TASK
	//arg.add("reach_1_flag"); // reach flag task
	arg.add("follow_the_line");
	c->sendCommand("LIST_ENVIRONMENTS",arg);
	do{
		c->waitResponse(strResponse, arguments);
	
	}while(*strResponse=="ENVIRONMENT");

	//SELECT Environment
	//arg.add("SingleRoom");
	arg.add("Line");
	c->sendCommand("INITIALIZE_TASK",arg);
	c->waitResponse(strResponse, arguments);
	c->waitResponse(strResponse, arguments);
	c->waitResponse(strResponse, arguments);

	c->sendCommand("BEGIN_TASK_SETUP",empty);
	c->waitResponse(strResponse, arguments);



	c->sendCommand("END_TASK_SETUP",empty);
	c->waitResponse(strResponse, suggestedAction);
	c->waitResponse(strResponse, arguments);
	c->waitResponse(strResponse, arguments);


	/////////////////////////////////
	/////START PLAYING///////////////
	/////////////////////////////////



	arg.clear();
	arg.add("main");
	for(int i=0; i<1000;i++)
	{	
		bool predictFlag =0;
		counter=0;
		while(*strResponse!= "FINISHED" && counter < timeOut && *strResponse!= "FAILED")
		{
			
			c->sendCommand("GET_VIEW",arg);
			c->waitResponse(strResponse, arguments);
			c->waitData(data, 8);
			c->waitData(data, datasize);

			//writeDataset(&dataset,&target,data,suggestedAction->getString(0),datasize);
			//writeImage(data,320,240);
			/*if(suggestedAction->getString(0) == "TURN_RIGHT" && rightcounter<classlimit && rightcounter<= max(leftcounter,forwardcounter))
			{
				writeflag=1;
				rightcounter ++;
			}
			if(suggestedAction->getString(0) == "TURN_LEFT" && leftcounter<classlimit && leftcounter<= max(rightcounter,forwardcounter))
			{
				writeflag=1;
				leftcounter ++;
			}
			if(suggestedAction->getString(0) == "GO_FORWARD" && forwardcounter<classlimit && forwardcounter<= max(leftcounter,rightcounter))
			{
				writeflag=1;
				forwardcounter ++;
			}

			if(writeflag==1)
			{
				writeDataset(&dataset,&target,data,suggestedAction->getString(0),datasize);
			}
			writeflag=0;*/
///////////////////////////////////////////////////////////////////////////////////////////////
//				MAKE PREDICTION
///////////////////////////////////////////////////////////////////////////////////////////////	
/*
//fill message
float average;

StringBuffer s;
Writer<StringBuffer> writer(s);

writer.StartObject();
writer.String("A");
writer.StartArray();

for(int i=0;i<(datasize);i+=3)
{
	average = 0.2126*(float)data[i] + 0.7152*(float)data[i+1] + 0.0722*(float)data[i+2];
	writer.Double((int)average);
}
writer.EndArray();
writer.EndObject();

string sssss= s.GetString();
int msgsize=sssss.length();
cout<<"msgsize = "<<msgsize<<endl;
//cout<<"s.GetString() = "<<s.GetString()<<endl;
// send message
zmq::message_t request (msgsize);
memcpy ((void *) request.data (), s.GetString(), msgsize);


socket.send (request);

//get reply
zmq::message_t reply;
socket.recv (&reply);

string prediction = string(static_cast<char*>(reply.data()), reply.size());
actions.clear();
actions.add(prediction);

if(prediction!=suggestedAction->getString(0))
{
	if (predictFlag ==1)
	{
		predictFlag=0;
	}
	else
	{
		predictFlag=1;		
	}


}


*/

///////////////////////////////////////////////////////////////////////////////////////////////		
			/*if (predictFlag == 0)
			{
				c->sendCommand("ACTION",*suggestedAction); // if following teacher
			}	
			else
			{
				c->sendCommand("ACTION",actions); // if predicting
			}*/
			c->sendCommand("ACTION",*suggestedAction); // if following teacher
			c->waitResponse(strResponse, reward);
			c->waitResponse(strResponse, suggestedAction);
			c->waitResponse(strResponse, arguments);
			c->waitResponse(strResponse, arguments);

			counter++;
		}

		if(*strResponse=="FINISHED")
			numFinish++;		

		c->sendCommand("RESET_TASK",empty);
		c->waitResponse(strResponse, suggestedAction);
		c->waitResponse(strResponse, arguments);
		c->waitResponse(strResponse, arguments);

		cout<<"round  = "<<i<<endl;
		cout<<"numfinished  = "<<numFinish<<endl;
	}
	
	dataset.close();
	target.close();
	c->sendCommand("DONE",empty);
	c->waitResponse(strResponse, arguments);

	cout<<"score = "<<numFinish<<endl;

	return 0;
}
