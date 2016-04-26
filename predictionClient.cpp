#include <stdio.h>
#include <iostream>
#include <fstream>
#include <zmq.hpp>
#include <zmq_utils.h>
#include <string>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <mash-network/client.h>
//#include <Image.h>
//#include <svm.h>
#include <sstream>

using namespace std;
using namespace Mash;
using namespace rapidjson;




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
		average = 0.2126*(float)data[i] + 0.7152*(float)data[i+1] + 0.0722*(float)data[i+2]; // get luminance map from 3 channel image
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
	const int width = 120; //width of frame
	const int height = 90; //height of frame
	const int datasize = width*height*3; // 3 channel image size
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
	string task = "reach_1_flag"; // reach flag task
	//string task = "follow_the_line"; // follow line task
	string environment = "SingleRoom"; // reach flag task
	//string environment = "line"; // follow line task


///////////Connect to Prediction Server/////////////
	zmq::context_t context (1);
	zmq::socket_t socket (context, ZMQ_REQ);


	//socket.bind ("tcp://*:5555"); // if server
	socket.connect ("tcp://localhost:5555"); // if client

////////////////////////////////////////////////////////


	//Connect to mash-simulator Server/////////////

	string* strResponse = new string;
	ArgumentsList* arguments = new ArgumentsList();
	ArgumentsList* reward = new ArgumentsList(0);
	ArgumentsList* suggestedAction = new ArgumentsList();

	c->connect("127.0.0.1",11200);

	c->sendCommand("STATUS",empty);
	c->waitResponse(strResponse, arguments);
	
	c->sendCommand("USE_GLOBAL_SEED",12345); 
	c->waitResponse(strResponse, arguments);	
	
	//Get a list of tasks	
	c->sendCommand("LIST_GOALS",empty);
	do{
		c->waitResponse(strResponse, arguments);
	
	}while(*strResponse=="GOAL");

	//SELECT TASK
	arg.add(task);

	//Get a list of Environments
	c->sendCommand("LIST_ENVIRONMENTS",arg);
	do{
		c->waitResponse(strResponse, arguments);
	
	}while(*strResponse=="ENVIRONMENT");

	//SELECT Environment
	arg.add(environment);

	//Setup the task
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

	cout<<"in start playing"<<endl;

	arg.clear();
	arg.add("main");
	for(int i=0; i<1000;i++) //number of rounds
	{	
		cout<<"in for"<<endl;
		bool predictFlag =1; // decide to use ground truth or agent predictions
		counter=0; // number of frames elapsed in a round
		while(*strResponse!= "FINISHED" && counter < timeOut && *strResponse!= "FAILED") // while the round hasn't failed or succeeded yet, and time hasn't run out
		{
			cout<<"in while"<<endl;	
			cout<<"strResponse  "<<*strResponse<<endl;
			c->sendCommand("GET_VIEW",arg); // get image frame from the simulator
			c->waitResponse(strResponse, arguments);
			c->waitData(data, 8);
			c->waitData(data, datasize);

			//writeDataset(&dataset,&target,data,suggestedAction->getString(0),datasize);


///////////////////////////////////////////////////////////////////////////////////////////////
//				MAKE PREDICTION
///////////////////////////////////////////////////////////////////////////////////////////////	
			cout<<"in make prediction"<<endl;
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
				//average=0;
				writer.Double((int)average);
			}
			writer.EndArray();
			writer.EndObject();


			string sssss= s.GetString();
			int msgsize=sssss.length();
			//cout<<"msgsize = "<<msgsize<<endl;
			//cout<<"s.GetString() = "<<s.GetString()<<endl;
			// send message
			zmq::message_t request (msgsize);
			memcpy ((void *) request.data (), s.GetString(), msgsize);


			socket.send (request);

			//get reply
			zmq::message_t reply;
			socket.recv (&reply);

			string prediction = string(static_cast<char*>(reply.data()), reply.size());
			cout<<"prediction = "<<prediction<<endl;
			actions.clear();
			actions.add(prediction);

			/*if(prediction!=suggestedAction->getString(0))
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
			cout<<"in send predciction"<<endl;
			// send an action to the mash-simulator server
			if (predictFlag == 0)
			{
				c->sendCommand("ACTION",*suggestedAction); // if following teacher
			}	
			else
			{
				c->sendCommand("ACTION",actions); // if predicting
			}
			cout<<"in prediction"<<endl;
			c->waitResponse(strResponse, reward);
			cout<<"in prediction"<<endl;
			c->waitResponse(strResponse, suggestedAction);
			cout<<"in prediction"<<endl;
			c->waitResponse(strResponse, arguments);
			cout<<"in prediction"<<endl;
			c->waitResponse(strResponse, arguments);
			cout<<"in prediction"<<endl;

///////////////////////////////////////////////////////////////////////////////////////////////
//       			 send reward to DQN
			cout<<"in send reward"<<endl;
			string reward_string = reward->getString(0);

			StringBuffer rsb;
			Writer<StringBuffer> rwriter(rsb);
			rwriter.StartObject();
			rwriter.String("A");
			rwriter.StartArray();

			rwriter.String(reward_string);
			
			rwriter.EndArray();
			rwriter.EndObject();

			string rmessage= rsb.GetString();
								
			zmq::message_t reward_msg (rmessage.length());
			memcpy (reward_msg.data (), rsb.GetString(), rmessage.length());
			cout<< "reward is "<< reward_string<<endl;
			socket.send(reward_msg);

			socket.recv (&reply); // dummy reply

////////////////////////////////////////////////////////////////////////////////////////////////

			counter++;
		}//End of one round
		cout<<"out of while"<<endl;
		if(*strResponse=="FINISHED")
			numFinish++;// increment number of succesful rounds		

		c->sendCommand("RESET_TASK",empty); // initialize a new round
		c->waitResponse(strResponse, suggestedAction);
		c->waitResponse(strResponse, arguments);
		c->waitResponse(strResponse, arguments);

		cout<<"round  = "<<i<<endl;
		cout<<"numfinished  = "<<numFinish<<endl;
	}// end of all rounds
	cout<<"out of for while"<<endl;
	dataset.close();
	target.close();
	c->sendCommand("DONE",empty); // shut down connection to server
	c->waitResponse(strResponse, arguments);

	cout<<"score = "<<numFinish<<endl;

	return 0;
}
