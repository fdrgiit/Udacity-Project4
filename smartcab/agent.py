import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.e = 0.1     
        self.numberOfTrials = 100
        self.ke = -0.008 
        self.qInit = 2 
        self.alphalearn = 0.3     
        self.gammaDiscount = 0.99     

        self.lastState = None
        self.lastAction = None
        self.reward = 0
        self.Q = {}
        self.t = 0          
        self.track_e = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.lastState = None
        self.lastAction = None
        self.reward = 0
        self.t += 1
        self.track_e.append(100 * min(1, max(0, self.e + self.ke * self.t)))

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # TODO: Learn policy based on current & previous state, previous action, reward got during state transition
        self.learnDrive()
        
        # TODO: Select action according to your policy and current state
        epsilon = self.e + self.ke * self.t
        draw = random.random()
        if draw < epsilon:
            return random.choice(Environment.valid_actions)
        else:
            maxQ = -9999
            maxAction = None
            for i_action in Environment.valid_actions:
                curQ = self.drive((self.state, i_action))
                if maxQ < curQ :
                    maxQ = curQ
                    maxAction = i_action
                    
        self.action = maxAction

        # Execute action and get reward
        self.reward = self.env.act(self, self.action)
        self.lastState = self.state
        self.lastAction = self.action
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, self.reward)  # [debug]

    def learnDrive(self):
        lastQ = self.drive((self.lastState, self.lastAction))
        
        maxQ = 0
        for i_action in Environment.valid_actions:
            maxQ = max(maxQ, self.drive((self.state, i_action)))

        self.Q[(self.lastState, self.lastAction)] = lastQ * (1 - self.alphalearn) + self.alphalearn * (self.reward + self.gammaDiscount * maxQ)

    def drive(self, state_action_pair):
        if state_action_pair == None: return self.qInit
        return self.Q.setdefault(state_action_pair, self.qInit) # if not contain the key, init it with self.qInit

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
