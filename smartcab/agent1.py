import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = defaultdict(list)
        self.alpha = .3
        self.gamma = .99
        self.stateAction = None
        self.epsilon = .1
        self.penalty  = 0
        self.oldPenalty = None

    def reset(self, destination=None):
        self.stateAction = None
        self.planner.route_to(destination)
        print 'penalty = {}'.format(self.penalty if self.oldPenalty is None else (self.penalty-self.oldPenalty)) 
        self.oldPenalty = self.penalty
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = [inputs['light'],inputs['oncoming'],inputs['left'], self.next_waypoint]

        # TODO: Select action according to your policy

        action = None
        maxQ = 0
        if random.random() < self.epsilon:
            action = random.choice([None, 'forward', 'left', 'right']) 
        else:
            for k,v in self.Q.items():
                if k[0:3] == self.state and v > maxQ:
                    action = k[4]
                    maxQ = v
            if action == None:
                action = random.choice([None, 'forward', 'left', 'right']) 


        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0: 
            self.penalty += 1
            # print 'violation', self.penalty


        # TODO: Learn policy based on state, action, reward
        if self.stateAction is not None:
            if self.Q[tuple(self.stateAction)]:
                self.Q[tuple(self.stateAction)] = self.Q[tuple(self.stateAction)] + self.alpha * (reward + self.gamma*maxQ - self.Q[tuple(self.stateAction)])
            else:
                self.Q[tuple(self.stateAction)] = self.alpha * (reward + self.gamma*maxQ )
        else:
            print 'Q learning start'



        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        self.stateAction = [inputs['light'],inputs['oncoming'],inputs['left'], self.next_waypoint, action]

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
