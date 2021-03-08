#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

import dotenv


from dotenv import load_dotenv, set_key
from microsoft_bonsai_api.simulator.client import BonsaiClient, BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import (
    SimulatorInterface,
    SimulatorState,
    SimulatorSessionResponse,
)
from azure.core.exceptions import HttpResponseError
from distutils.util import strtobool

#from policies import coast, random_policy
#from sim import cartpole


# In[2]:


import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#from unityagents import UnityEnvironment
import numpy as np
from DeepQN_Agent import Agent


# In[3]:


from sim import unityagents

#from sim_docker import unityagents


# In[4]:


from sim.unityagents import UnityEnvironment



# In[5]:


import numpy as np
import random
from collections import namedtuple ,deque
from NN_Model import DeepQNetModel

import torch
import torch.nn.functional as F
import torch.optim as optim


# In[6]:


# dir_path = os.path.dirname(os.path.realpath(__file__))
log_path = "logs"
default_config = {"eps_start" : 1.0 ,"eps_end":0.01,"eps_decay" :0.995}
# BUFFER_SIZE = int(1e5)  # replay buffer size
# BATCH_SIZE = 64         # minibatch size


# In[7]:


import random
from typing import Dict
import requests


def random_policy(state):
    """
    Ignore the state, move randomly.
    """
    print("RANDOM POLICY" )
    action = {"command": random.choice([0, 1 , 2, 3])}
    return action


def coast(state):
    """
    Ignore the state, go right.
    """
    
    print("COAST :::: " )
    
    action = {"command": 1}
    return action


def brain_policy(
    state: Dict[str, float], exported_brain_url: str = "http://localhost:5000"
):
    print("EXTERNAL BRAIN POLICY" )
    prediction_endpoint = f"{exported_brain_url}/v1/prediction"
    response = requests.get(prediction_endpoint, json=state)

    return response.json()


POLICIES = {"random": random_policy, "coast": coast}


# In[8]:


class TemplateSimulatorSession:
    def __init__(
        self,
        render: bool = False,
        env_name: str = "",
        log_data: bool = False,
        log_file: str = None,
    ):
        """Simulator Interface with the Bonsai Platform
        Parameters
        ----------
        render : bool, optional
            Whether to visualize episodes during training, by default False
        env_name : str, optional
            Name of simulator interface, by default "Cartpole"
        log_data: bool, optional
            Whether to log data, by default False
        log_file : str, optional
            where to log data, by default None
        """
        
        
        
        self.env = UnityEnvironment(file_name="sim/Banana_Windows_x86_64/Banana_Windows_x86_64/Banana.exe")
        
        #self.env = UnityEnvironment(file_name="sim/Banana_Windows_x86_64/Banana_Windows_x86_64/Banana")
        
        
        #self.env = UnityEnvironment(file_name="sim/Banana_Windows_x86_64/Banana_Windows_x86_64/bonsai_env.x86_64")
        
        
        #self.env = UnityEnvironment(file_name="Banana.exe")
        
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.env_info = self.env.reset(train_mode=False)[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state = self.env_info.vector_observations[0]
        
        print("self.state :::: " +str(self.state)) 
        self.state_size = len(self.state)
        
        self.simulator = Agent(state_size=self.state_size, action_size=self.action_size, seed=0)
        self.count_view = False
        self.env_name = self.brain_name
        self.render = render
        self.log_data = log_data
        
        if not log_file:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_file = current_time + "_" + env_name + "_log.csv"
            log_file = os.path.join(log_path, log_file)
            logs_directory = pathlib.Path(log_file).parent.absolute()
            if not pathlib.Path(logs_directory).exists():
                print(
                    "Directory does not exist at {0}, creating now...".format(
                        str(logs_directory)
                    )
                )
                logs_directory.mkdir(parents=True, exist_ok=True)
        self.log_file = os.path.join(log_path, log_file)
        
    def load_JSON_Config(self):
        with open("BananaCollector_description.json") as file:
             interface = json.load(file)
             
        return interface
         
        
    def get_state(self) -> Dict[str, float]:
        """Extract current states from the simulator
        Returns
        -------
        Dict[str, float]
            Returns float of current values from the simulator
        """
        #return Agent.ReplayBuffer.experience.state
        
        print("get state ::: " +str(self.env_info.vector_observations[0]))
        
        
        interface = self.load_JSON_Config()
        
        
        State_Names =  []

        for listietme in interface['description']['state']['fields']:
            #print(listietme["name"])
            State_Names.append(listietme["name"])
            
            
        StateDict= {}
        
        StateVals = list(self.env_info.vector_observations[0])
        
        for i in range(len(list(self.env_info.vector_observations[0]))):
            StateVals[i] = float(StateVals[i])
            #StateVals[i] = int(StateVals[i])
             

        for i in range(len(State_Names)):
            StateDict[State_Names[i]] = StateVals[i]
        
        
    
        #return self.env_info.vector_observations[0]

        #return str(self.env_info.vector_observations[0])        
        
        return StateDict
    
    def halted(self) -> bool:
        """Halt current episode. Note, this should only be called if the simulator has reached an unexpected state.
        Returns
        -------
        bool
            Whether to terminate current episode
        """
        return False
    
    def episode_start(self, config: Dict = None) -> None:
        """Initialize simulator environment using scenario paramters from inkling. Note, `simulator.reset()` initializes the simulator parameters for initial positions and velocities of the cart and pole using a random sampler. See the source for details.
        Parameters
        ----------
        config : Dict, optional
            masspole and length parameters to initialize, by default None
        """

#         if "length" in config.keys():
#             self.simulator.length = config["length"]
#         if "masspole" in config.keys():
#             self.simulator.masspole = config["masspole"]
        #self.simulator.reset(train_mode=False)[brain_name]
        env_info = self.env.reset(train_mode=False)[self.brain_name] # reset the environment
        if config is None:
            config = default_config
        self.config = config
        
        
    def log_iterations(self, state, action, episode: int = 0, iteration: int = 1):
        """Log iterations during training to a CSV.
        Parameters
        ----------
        state : Dict
        action : Dict
        episode : int, optional
        iteration : int, optional
        """

        import pandas as pd

        def add_prefixes(d, prefix: str):
            #print(d)
            return {f"{prefix}_{k}": v for k, v in d.items()}
        

        with open("BananaCollector_description.json") as file:
             interface = json.load(file)
       
                
        #State_Names =  []

        #for listietme in interface['description']['state']['fields']:
            #print(listietme["name"])
            #State_Names.append(listietme["name"])
            
        #print(State_Names)
        #print(len(State_Names))
        
        #print(len(list(state)))
        
        #print(type(action))
        
        #print(action)
        
        #StateDict= {}

        #for i in range(len(State_Names)):
            #StateDict[State_Names[i]] = list(state)[i]
        
        #print(StateDict)
        
        actionlist = []




        for listietme in interface['description']['action']['fields']:

            #print(listietme['type']['forward'])
            #print(type(listietme['type']['forward']))
            actionlist.append(listietme['type']['forward'])
            actionlist.append(listietme['type']['backward'])
            actionlist.append(listietme['type']['left'])
            actionlist.append(listietme['type']['right'])
            
            
        #print(actionlist)
        
        actionnames =[]

        for listietme in interface['description']['action']['fields']:
            for keys in listietme['type']:
                #print(keys)
                actionnames.append(keys)

        #print(actionnames)


        actionnames.pop(0)


        #print(actionnames)


        actionnames = actionnames[:-1]



        #print(actionnames)
        
        
        ActionDict= {}

        for i in range(len(actionnames)):
            ActionDict[actionnames[i]] = actionlist[i]
            
            
        #print(ActionDict)
        
        
        action = list(action)
        
        
        Action_Selected_Dict={}


        for key, val in ActionDict.items():
            #print(key,val)
            if(val == action[0]):
                Action_Selected_Dict[key]=val
            
        #print(Action_Selected_Dict)
        
        #print(self.log_file)

        
        state = add_prefixes(state, "state")
        #state = add_prefixes(StateDict, "state")
        #action = add_prefixes(action, "action")
        action = add_prefixes(Action_Selected_Dict, "action")
        config = add_prefixes(self.config, "config")
        data = {**state, **action, **config}
        data["episode"] = episode
        data["iteration"] = iteration
        log_df = pd.DataFrame(data, index=[0])

        if os.path.exists(self.log_file):
            log_df.to_csv(
                path_or_buf=self.log_file, mode="a", header=False, index=False
            )
        else:
            #log_df.to_csv(path_or_buf=self.log_file, mode="w", header=True, index=False)
            
            log_df.to_csv(self.log_file)
            
            
    def episode_step(self, action: Dict):
    #def episode_step(self, action):
        """Step through the environment for a single iteration.
        Parameters
        ----------
        action : Dict
            An action to take to modulate environment.
        """
        #self.simulator.step(action["command"])
        
        #action = np.random.randint(self.action_size)
        #action =  self.simulator.act(state, 0.0)       # select an action
        
        
        print("EPISODE STEP ::: ")
        
        print("action debug: ")
        
        print(action[0])
        print(action)
        
        print("ACTION " +str( action))
        
        
        
        
        
        
        env_info = self.env.step(int(action[0]))[self.brain_name] 
        
#         if self.render:
#             self.sim_render()        
            
            
        
        


# In[9]:


def env_setup():
    """Helper function to setup connection with Project Bonsai
    Returns
    -------
    Tuple
        workspace, and access_key
    """

    load_dotenv(verbose=True)
    workspace = os.getenv("725a8f15-dd22-49b2-8dac-b80bc0e0692f")
    access_key = os.getenv("ZTA3NWYyYjVkZWE5NGM0ZTgyM2RmMTM0Y2I4YjM4M2U6OGU4ZjIwYzgtMjY1Ni00Y2FhLWJhNjEtMmI1YjM0NTFjYWNl")

    env_file_exists = os.path.exists(".env")
    if not env_file_exists:
        open(".env", "a").close()

    if not all([env_file_exists, workspace]):
        workspace = input("Please enter your workspace id: ")
        set_key(".env", "725a8f15-dd22-49b2-8dac-b80bc0e0692f", workspace)
    if not all([env_file_exists, access_key]):
        access_key = input("Please enter your access key: ")
        #set_key(".env", "MTc2MDQ4M2FkM2JmNGJlMDkzM2JhMjZhY2I2MjJhOTE6ZGNiZGViZWYtMzE2Yy00YjQxLTlmMjgtMjM4OTkyZTViNTI3", access_key)
        set_key(".env", "ZTA3NWYyYjVkZWE5NGM0ZTgyM2RmMTM0Y2I4YjM4M2U6OGU4ZjIwYzgtMjY1Ni00Y2FhLWJhNjEtMmI1YjM0NTFjYWNl", access_key)
        
        
    load_dotenv(verbose=True, override=True)
    workspace = os.getenv("725a8f15-dd22-49b2-8dac-b80bc0e0692f")
    #access_key = os.getenv("MTc2MDQ4M2FkM2JmNGJlMDkzM2JhMjZhY2I2MjJhOTE6ZGNiZGViZWYtMzE2Yy00YjQxLTlmMjgtMjM4OTkyZTViNTI3")
    access_key = os.getenv("ZTA3NWYyYjVkZWE5NGM0ZTgyM2RmMTM0Y2I4YjM4M2U6OGU4ZjIwYzgtMjY1Ni00Y2FhLWJhNjEtMmI1YjM0NTFjYWNl")

    return workspace, access_key


# In[10]:


def transform_state(sim_state):
    """
    Convert
    {'x_position': 0.008984250082219904, 'x_velocity': -0.026317885259067864, 'angle_position': -0.007198829694026056, 'angle_velocity': -0.03567818795116845}
    from the sim into what my brain expects
    ['cart_position', 'cart_velocity', 'pole_angle', 'pole_angular_velocity'] 
    """
    s = sim_state
    return {
        "cart_position": s["x_position"],
        "cart_velocity": s["x_velocity"],
        "pole_angle": s["angle_position"],
        "pole_angular_velocity": s["angle_velocity"],
    }


def transform_action(action):
    """
    Implementing the selector logic here...
    expecting action to have fields command_forward , command_backward , command_left ,command_right, for the four subconcepts
    """
    # Let's try command left for now
    return {"command": action["command_right"]}


# In[11]:


def test_policy(
    policy,
    num_episodes: int = 10,
    #num_episodes: int = 1,
    render: bool = True,
    num_iterations: int = 200,
    log_iterations: bool = False,
    policy_name:str = "random"
):
    """Test a policy using random actions over a fixed number of episodes
    Parameters
    ----------
    num_episodes : int, optional
        number of iterations to run, by default 10
    """

    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_file_name = current_time + "_" + policy_name + "_log.csv"
    sim = TemplateSimulatorSession(
        render=render, log_data=log_iterations, log_file=log_file_name
    )
    # test_config = {"length": 1.5}
    for episode in range(num_episodes):
        iteration = 0
        terminal = False
        sim_state = sim.episode_start(config=default_config)
        sim_state = sim.get_state()
        
        print("SIM STATE TEST POLIXY  : " +str(sim_state))
        while not terminal:
            #action = policy(sim_state)
            action =np.random.randint(sim.action_size),
            print("action ::" + str(action) + "datadtype of action::" + str(type(action)))
            sim.episode_step(action)
            sim_state = sim.get_state()
            print("SIM STATE TEST POLIXY LOOOP  : " +str(sim_state))
            if log_iterations:
                sim.log_iterations(sim_state, action, episode, iteration)
            print(f"Running iteration #{iteration} for episode #{episode}")
            print(f"Observations: {sim_state}")
            print("Number of Iterations" +str(iteration))
            print("NUMBER OF EPISODES" +str(episode))
            
            iteration += 1
            terminal = iteration >= num_iterations
            
        print("NUMBER OF EPISODES : " +str(episode))

    return sim


# In[12]:


def main(
    render: bool = False, log_iterations: bool = False, config_setup: bool = False
):
    """Main entrypoint for running simulator connections
    Parameters
    ----------
    render : bool, optional
        visualize steps in environment, by default True, by default False
    log_iterations: bool, optional
        log iterations during training to a CSV file
    """

    # workspace environment variables
    if config_setup:
        workspace, access_key = env_setup()
        print("workspace :: " +str(workspace))
        print("access_key :: " +str(access_key))
        
        load_dotenv(verbose=True, override=True)

    # Grab standardized way to interact with sim API
    sim = TemplateSimulatorSession(render=render, log_data=log_iterations )

    # Configure client to interact with Bonsai service
    config_client = BonsaiClientConfig()
    #config_client = BonsaiClientConfig(workspace: str = '725a8f15-dd22-49b2-8dac-b80bc0e0692f', access_key: str = 'MTc2MDQ4M2FkM2JmNGJlMDkzM2JhMjZhY2I2MjJhOTE6ZGNiZGViZWYtMzE2Yy00YjQxLTlmMjgtMjM4OTkyZTViNTI3', enable_logging=True)
    client = BonsaiClient(config_client)
    
    print("CONFIG CLIENT ACCESS KEY :::: " +str(config_client.access_key))
    
    print("CONFIG CLIENT workspace ID :::: " +str(config_client.workspace))

    # # Load json file as simulator integration config type file
    with open("BananaCollector_description.json") as file:
        interface = json.load(file)
    #print(type(interface))
    #print(interface)

    # Create simulator session and init sequence id
    registration_info = SimulatorInterface(
        name=sim.env_name,
        timeout=interface["timeout"],
        simulator_context=config_client.simulator_context,
        description=interface["description"],
    )
    
    #print(registration_info)
    #CreateSession(registration_info,config_client,client)
    
    
   
    
    #print("registered_session :" +str(registration_info))
    #print("sequence id : " +str(sequence_id))
    #print("session id : " +str(registered_session.session_id))
    


# In[14]:


    def CreateSession(
            registration_info: SimulatorInterface, config_client: BonsaiClientConfig
        ):
        """Creates a new Simulator Session and returns new session, sequenceId
        """
        
        #print("registration_info:" +str(registration_info))
        #print("config_client:" +str(config_client))
        #print("client : "+str(client))

        try:
            print(
                "config: {}, {}".format(config_client.server, config_client.workspace)
            )
            registered_session: SimulatorSessionResponse = client.session.create(
                workspace_name=config_client.workspace, body=registration_info
            )
            print("Registered simulator. {}".format(registered_session.session_id))

            return registered_session, 1
        except HttpResponseError as ex:
            print(
                "HttpResponseError in Registering session: StatusCode: {}, Error: {}, Exception: {}".format(
                    ex.status_code, ex.error.message, ex ,ex.reason , ex.model ,ex.response , ex.error
                )
            )
            raise ex
            
        except Exception as ex:
            print(
                "UnExpected error: {}, Most likely, it's some network connectivity issue, make sure you are able to reach bonsai platform from your network.".format(
                    ex
                )
            )
            raise ex

    registered_session, sequence_id = CreateSession(registration_info, config_client)
    
    print("registered_session :" +str(registered_session))
    print("sequence id : " +str(sequence_id))
    print("session id : " +str(registered_session.session_id))
    episode = 0
    iteration = 0

    try:
        while True:
            # Advance by the new state depending on the event type
            # TODO: it's risky not doing doing `get_state` without first initializing the sim
            
            print("TRUE LOOP CREATE SESSION :::::::" ) 
            sim_state = SimulatorState(
                sequence_id=sequence_id, state=sim.get_state(), halted=sim.halted(),
            )
            print("CREATE SESSION SIM STATE ::: "+ str(sim_state))
            try:
                event = client.session.advance(
                    workspace_name=config_client.workspace,
                    session_id=registered_session.session_id,
                    body=sim_state,
                )
                sequence_id = event.sequence_id
                print(
                    "[{}] Last Event: {}".format(time.strftime("%H:%M:%S"), event.type)
                )
            except HttpResponseError as ex:
                print(
                    "HttpResponseError in Advance: StatusCode: {}, Error: {}, Exception: {}".format(
                        ex.status_code, ex.error.message, ex
                    )
                )
                # This can happen in network connectivity issue, though SDK has retry logic, but even after that request may fail,
                # if your network has some issue, or sim session at platform is going away..
                # So let's re-register sim-session and get a new session and continue iterating. :-)
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            except Exception as err:
                print("Unexpected error in Advance: {}".format(err))
                # Ideally this shouldn't happen, but for very long-running sims It can happen with various reasons, let's re-register sim & Move on.
                # If possible try to notify Bonsai team to see, if this is platform issue and can be fixed.
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue

            # Event loop
            if event.type == "Idle":
                time.sleep(event.idle.callback_time)
                print("Idling...")
            elif event.type == "EpisodeStart":
                print(event.episode_start.config)
                sim.episode_start(event.episode_start.config)
                
                print(" EVENT EPISODE START CONFIG" +str(event.episode_start.config))
                episode += 1
            elif event.type == "EpisodeStep":
            
                print(" EVENT EPISODE STEP" ) 
            
            
                iteration += 1
                
                print( " EVENT EPISODE STEP ACTION ---------> "  +str (event.episode_step.action)) 
                
                action =np.random.randint(sim.action_size)
                
                action = (action,)
                
                print("datatype of action" + str(type(action)))
                
                print("action of event " +str(action))
                
                #sim.episode_step(event.episode_step.action)
                
                sim.episode_step(action)
                
                if sim.log_data:
                    sim.log_iterations(
                        episode=episode,
                        iteration=iteration,
                        state=sim.get_state(),
                        #action=event.episode_step.action,
                        action=action,
                    )
            elif event.type == "EpisodeFinish":
                print("Episode Finishing...")
                iteration = 0
            elif event.type == "Unregister":
                print(
                    "Simulator Session unregistered by platform because '{}', Registering again!".format(
                        event.unregister.details
                    )
                )
                registered_session, sequence_id = CreateSession(
                    registration_info, config_client
                )
                continue
            else:
                pass
    except KeyboardInterrupt:
        print("session id : " +str(registered_session.session_id))
        # Gracefully unregister with keyboard interrupt
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator.")
    except Exception as err:
        # Gracefully unregister for any other exceptions
        client.session.delete(
            workspace_name=config_client.workspace,
            session_id=registered_session.session_id,
        )
        print("Unregistered simulator because: {}".format(err))


# In[15]:


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Bonsai and Simulator Integration...")
    
    #parser.add_argument(
    #"--workspace-id", action="store_true", default="725a8f15-dd22-49b2-8dac-b80bc0e0692f", help="Simulator Workspace ID",
    #)
    
    #parser.add_argument(
    #"--accesskey", action="store_true", default="MTc2MDQ4M2FkM2JmNGJlMDkzM2JhMjZhY2I2MjJhOTE6ZGNiZGViZWYtMzE2Yy00YjQxLTlmMjgtMjM4OTkyZTViNTI3", help="Simulator Access Key",
    #)
    
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render training episodes",
    )
    parser.add_argument(
        "--log-iterations",
        action="store_true",
        default=False,
        help="Log iterations during training",
    )
    parser.add_argument(
        "--config-setup",
        action="store_true",
        default=False,
        help="Use a local environment file to setup access keys and workspace ids",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--test-random",
        action="store_true",
        help="Run simulator locally with a random policy, without connecting to platform",
    )

    group.add_argument(
        "--test-exported",
        type=int,
        const=5000,  # if arg is passed with no PORT, use this
        nargs="?",
        metavar="PORT",
        help="Run simulator with an exported brain running on localhost:PORT (default 5000)",
    )

    args = parser.parse_args()
    
    print("args :" +str(args))

    if args.test_random:
        test_policy(
            render=args.render, log_iterations=args.log_iterations, policy=random_policy
        )
   
    elif args.test_exported:
        port = args.test_exported
        url = f"http://localhost:{port}"
        print(f"Connecting to exported brain running at {url}...")
        trained_brain_policy = partial(brain_policy, exported_brain_url=url)
        test_policy(
            render=args.render,
            log_iterations=args.log_iterations,
            policy=trained_brain_policy,
            policy_name="exported"
        )
    else:
        main(
            config_setup=args.config_setup,
            render=args.render,
            log_iterations=args.log_iterations,
        )


# In[ ]:




