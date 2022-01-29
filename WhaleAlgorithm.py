import random as rd
import numpy as np
import argparse
import sys
import os
import glob
import copy
import matplotlib.pyplot as plt
from datetime import datetime

class Agent:
    def __init__(self, n_items):
        self.items = np.array([float(rd.randint(0, 1)) for _ in range(n_items)])
        self.fitness = 0

class Items:
    def __init__(self, weights, values):
        self.weights = weights
        self.values = values

# ----- Générer l'état initial
def initialize_population(n_pop, n_items):
    return [Agent(n_items) for _ in range(n_pop)]

#-----Calculer la condition physique
def calc_fitness(agent, items, n_items, max_weight):
    value = 0
    weight = 0
    
    agent_items = agent.items
    weight_arr = items.weights
    value_arr = items.values
    
    for i in range(n_items):
        if agent_items[i] > 0.5:
            value += value_arr[i]
            weight += weight_arr[i]
            
    fitness = 0
    
    if weight > max_weight:
        fitness = -1
    else:
        fitness = value
    
    return fitness

#-----Recherchez la personne avec la plus grande qualité d'ajustement
def search_best_agent(X_agents):
    return max(X_agents, key=lambda t: t.fitness)


#----- Partie constante mise à jour (a, A, C, l, p)
def update_const(a, a_step, n_items):
    #a, A, C
    a -= a_step
    r = np.random.uniform(0.0, 1.0, size=n_items)
    A = 2.0 * np.multiply(a, r) - a
    C = 2.0 * r
    #lMettre à jour 
    l = rd.uniform(-1.0, 1.0)
    #pMettre à jour 
    p = rd.uniform(0.0, 1.0)
    
    return a, A, C, l, p

def calc_encircle(sol, best_sol, A, C):
    D = np.linalg.norm(np.multiply(C, best_sol) - sol)
    return best_sol - np.multiply(A, D)

def calc_search(sol, rand_sol, A, C):
    D = np.linalg.norm(np.multiply(C, rand_sol) - sol)
    return rand_sol - np.multiply(A, D)

def calc_attack(sol, best_sol, l, b):
    D = np.linalg.norm(best_sol - sol)
    return np.multiply(np.multiply(D, np.exp(b * l)), np.cos(2.0*np.pi*l)) + best_sol


def adjust_agent(X_agents):
    for i in range(len(X_agents)):
        agent = X_agents[i]
        for j in range(len(agent.items)):
            if agent.items[j] < -1.0:
                agent.items[j] = -1.0
            elif agent.items[j] > 2.0:
                agent.items[j] = 2.0
        X_agents[i] = agent

def parse_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nsols", type=int, default=50, dest='nsols', help='number of solutions per generation, default: 50')
    parser.add_argument("-iter", type=int, default=50, dest='iter', help='number of iterations')
    parser.add_argument("-a", type=float, default=2.0, dest='a', help='woa algorithm specific parameter, controls search spread default: 2.0')
    parser.add_argument("-b", type=float, default=0.5, dest='b', help='woa algorithm specific parameter, controls spiral, default: 0.5')
    parser.add_argument("-p", type=str, default='./problems/07', dest='problem', help='Enter the directory containing the problem')

    args = parser.parse_args()
    return args


def load_problems(problems):
    print("____________________________________________")
    print("Now loading problems\n")
    if not os.path.exists(problems):
        raise ValueError("not exist such directory !")
    
    volume_file = glob.glob(problems + "/*_v.txt")
    volume_file = volume_file[0]
    if not os.path.exists(volume_file):
        raise ValueError("not exist volume_file !")
    
    weights_file = glob.glob(problems + "/*_w.txt")
    weights_file = weights_file[0]
    if not os.path.exists(weights_file):
        raise ValueError("not exist weights_file !")
    
    profits_file = glob.glob(problems + "/*_p.txt")
    profits_file = profits_file[0]
    if not os.path.exists(profits_file):
        raise ValueError("not exist profits_file !")
    
    solution_file = glob.glob(problems + "/*_s.txt")
    solution_file = solution_file[0]
    if not os.path.exists(solution_file):
        raise ValueError("not exist solution_file !")
    
    volume = 0
    weights = []
    profits = []
    solutions = []

    with open(volume_file, 'r') as fc:
        file_data = fc.readlines()
        for line in file_data:
            volume = int(line.rstrip())
    
    print("volume : " + str(volume) + "\n")
    
    with open(weights_file, 'r') as fw:
        file_data = fw.readlines()
        for line in file_data:
            weights.append(int(line.rstrip()))
    
    with open(profits_file, 'r') as fp:
        file_data = fp.readlines()
        for line in file_data:
            profits.append(int(line.rstrip()))
    
    with open(solution_file, 'r') as fs:
        file_data = fs.readlines()
        for line in file_data:
            solutions.append(int(line.rstrip()))
    
    print("items :")
    for i in range(1, len(weights) + 1):
        print("   " + str(i) + ". weight : " + str(weights[i-1]) +",  value : " +  str(profits[i-1]))
    print("")
    
    items = Items(weights, profits)
    
    solution = 0
    for i in range(len(weights)):
        if solutions[i] == 1:
            solution += profits[i]
    
    print("_______________")
    
    return  volume, items, len(weights), solution, solutions


def result(best, n_items, items, solution, solutions, volume, x, y1, y2, problems, nsols):
    print("__________________")
    print("Result")
    print("")
    nums = []
    weights = items.weights
    values = items.values
    labels = best.items
    total_weight = 0
    total_values = 0
    
    for i in range(n_items):
        if labels[i] > 0.5:
            nums.append(i)
            total_weight += weights[i]
            total_values += values[i]
    
    print("total weights : " + str(total_weight) + " (volume : " + str(volume) + "),   total values : " + str(total_values))
    for i in nums:
        print("   " + str(i + 1) + ". weight : " + str(weights[i]) + ",  value : " + str(values[i]))
    print("")

    print("Meilleur Solution : " + str(solution))
    for i in range(n_items):
        if solutions[i] == 1:
            print("   " + str(i + 1) + ". weight : " + str(weights[i]) + ",  value : " + str(values[i]))
    print("")
    
    fig = plt.figure()
    fig.subplots_adjust(left=0.2)
    plt.plot(x, y1, linestyle='solid', color='blue', label="best_fitness")
    plt.plot(x, y2, linestyle='dashdot', color='red', label='assumed_solution')
    plt.title("Best Fitness")
    plt.xlabel("iteration")
    plt.ylabel("best fitness")
    plt.ylim(min(y1) - 50, max(y2) + 50)
    plt.grid(linestyle='dotted')
    plt.legend(loc='lower right')
    
    date = datetime.now().strftime("%d_%H%M%S")
    
    fig.savefig(problems + "/iter"+ str(len(x)) +".jpg")


def main():
    args = parse_cl_args()
    nsols = args.nsols
    n_iter = args.iter
    problems = args.problem
    volume, items, n_items, solution, solutions= load_problems(problems)
    
    a = args.a
    a_step = a / n_iter / nsols
    a += a_step
    b = args.b
    x = [(i+1) for i in range(n_iter)]
    y1 = []
    y2 = [solution for _ in range(n_iter)]
    y_avg = []

    
    #Générer une génération précoce, Initialize the whales population Xi (i = 1, 2, ..., n)
    X = initialize_population(nsols, n_items)
    
    #Calculer la qualité de l'ajustement, ,  Calculate the fitness of each search agent
    for i in range(nsols):
        agent = X[i]
        agent.fitness = calc_fitness(agent, items, n_items, volume)
        X[i] = agent
    
   # Sortez l'agent avec la plus grande qualité d'ajustement
    x_star = search_best_agent(X)
    
   #Répéter entre les générations
    for t in range(n_iter):
        x_star_copy = copy.deepcopy(x_star)
        # Répéter dans 1 génération
        for i in range(nsols):
            a, A, C, l, p = update_const(a, a_step, n_items)
            agent = X[i]
            agent_tmp = copy.deepcopy(agent)

   #p between 0 and 1 is used to choose between the two mechanisms (shrinking encircling mechanism and the spiral-shaped path)
            if p < 0.5:

                if np.linalg.norm(A) < 1:
                   #Rapprochez-vous de la proie
                    x_star_tmp = copy.deepcopy(x_star)
                    agent.items = calc_encircle(agent_tmp.items, x_star_tmp.items, A, C)
                else:
                    #Trouver une proie
                    rand_sol = rd.choice(X)
                    agent.items = calc_search(agent_tmp.items, rand_sol.items, A, C)
            else:
               # tournoyer
                x_star_tmp = copy.deepcopy(x_star)
                agent.items = calc_attack(agent_tmp.items, x_star_tmp.items, l, b)
            X[i] = agent

       # Calcul de la qualité de l'ajustement
        for i in range(nsols):
            agent = X[i]
            agent.fitness = calc_fitness(agent, items, n_items, volume)
            X[i] = agent
        
        adjust_agent(X)

        #Repeindre x_star
        x_star = copy.deepcopy(x_star_copy)
        max_agent = search_best_agent(X)
        if x_star.fitness < max_agent.fitness:
            x_star = max_agent
            for i in range(n_items):
                if x_star.items[i] > 0.5:
                    x_star.items[i] = 1.0
                else:
                    x_star.items[i] = 0.0

        y1.append(x_star.fitness)

        if t < n_iter - 1:
            print("iteration : " + str(t + 1) + ", best_value : " + str(x_star.fitness))

   # Résultat de sortie
    result(x_star, n_items, items, solution, solutions, volume, x, y1, y2, problems, nsols)


if __name__ == '__main__':
    main()