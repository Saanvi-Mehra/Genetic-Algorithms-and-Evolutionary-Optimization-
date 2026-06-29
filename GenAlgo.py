# IMPORTS
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0', render_mode='human')


class NeuralNet:
    
   # Neural network to optimize the cartpole environment 
    
    def __init__(self, input_dim, hidden_dim, output_dim, test_run):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.test_run = test_run

    #helper functions
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def init_weights(self):
        input_weight = []
        input_bias = []

        hidden_weight = []
        out_weight = []

        input_nodes = 4

        for i in range(self.test_run):
            inp_w = np.random.rand(self.input_dim, input_nodes)
            input_weight.append(inp_w)
            inp_b = np.random.rand((input_nodes))
            input_bias.append(inp_b)
            hid_w = np.random.rand(input_nodes, self.hidden_dim)
            hidden_weight.append(hid_w)
            out_w = np.random.rand(self.hidden_dim, self.output_dim)
            out_weight.append(out_w)

        return [input_weight, input_bias, hidden_weight, out_weight]

    def forward_prop(self, obs, input_w, input_b, hidden_w, out_w):

        obs = obs/max(np.max(np.linalg.norm(obs)), 1)
        Ain = self.relu(obs@input_w + input_b.T)
        Ahid = self.relu(Ain@hidden_w)
        Zout = Ahid @ out_w
        A_out = self.relu(Zout)
        output = self.softmax(A_out)

        return np.argmax(output)

    def run_environment(self, input_w, input_b, hidden_w, out_w):
        obs, _ = env.reset()
        score = 0
        time_steps = 300
        for i in range(time_steps):
            action = self.forward_prop(obs, input_w, input_b, hidden_w, out_w)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            if terminated or truncated:
                break
        return score

    def run_test(self):
        generation = self.init_weights()
        input_w, input_b, hidden_w, out_w = generation
        scores = []
        for ep in range(self.test_run):
            score = self.run_environment(
                input_w[ep], input_b[ep], hidden_w[ep], out_w[ep])
            scores.append(score)
        return [generation, scores]


class GA:
    
    #Training neural net using genetic algorithm
    
    def __init__(self, init_weight_list, init_fitness_list, number_of_generation, pop_size, learner, mutation_rate=0.5):
        #initilize different parameters of the GA
        self.number_of_generation = number_of_generation
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.current_generation = init_weight_list
        self.current_fitness = init_fitness_list
        self.best_gen = []
        self.best_fitness = -1000
        self.fitness_list = []
        self.learner = learner

    def crossover(self, DNA_list):
     newDNAs = []
     while len(newDNAs) + len(DNA_list) < self.population_size:
        idxs = np.random.choice(len(DNA_list), 2, replace=False)
        p1, p2 = DNA_list[idxs[0]], DNA_list[idxs[1]]
        crossover_point = np.random.randint(1, len(p1)-1)
        child = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
        newDNAs.append(child)
     return newDNAs


    def mutation(self, DNA):
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.randn(len(DNA)) * 0.1
            DNA = DNA + mutation_vector
        return DNA

    def next_generation(self):
        index_good_fitness = np.argsort(self.current_fitness)[-2:]  # top 2 indices

        new_DNA_list = []
        new_fitness_list = []

        DNA_list = []
        for index in index_good_fitness:
            w1 = self.current_generation[0][index]
            dna_in_w = w1.reshape(w1.shape[1], -1)

            b1 = self.current_generation[1][index]
            dna_b1 = np.append(dna_in_w, b1)

            w2 = self.current_generation[2][index]
            dna_whid = w2.reshape(w2.shape[1], -1)
            dna_w2 = np.append(dna_b1, dna_whid)

            wh = self.current_generation[3][index]
            dna = np.append(dna_w2, wh)
            DNA_list.append(dna)

        new_DNA_list += DNA_list
        new_DNA_list += self.crossover(DNA_list)
        new_DNA_list = [self.mutation(dna) for dna in new_DNA_list]

        new_input_weight = []
        new_input_bias = []
        new_hidden_weight = []
        new_output_weight = []

        for newdna in new_DNA_list:
            newdna_in_w1 = np.array(
                newdna[:self.current_generation[0][0].size])
            new_in_w = np.reshape(
                newdna_in_w1, (-1, self.current_generation[0][0].shape[1]))
            new_input_weight.append(new_in_w)

            new_in_b = np.array(
                [newdna[newdna_in_w1.size:newdna_in_w1.size+self.current_generation[1][0].size]]).T  # bias
            new_input_bias.append(new_in_b)

            sh = newdna_in_w1.size + new_in_b.size
            newdna_in_w2 = np.array(
                [newdna[sh:sh+self.current_generation[2][0].size]])
            new_hid_w = np.reshape(
                newdna_in_w2, (-1, self.current_generation[2][0].shape[1]))
            new_hidden_weight.append(new_hid_w)

            sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
            new_out_w = np.array([newdna[sl:]]).T
            new_out_w = np.reshape(
                new_out_w, (-1, self.current_generation[3][0].shape[1]))
            new_output_weight.append(new_out_w)

            score = self.learner.run_environment(new_in_w, new_in_b, new_hid_w, new_out_w)
            new_fitness_list.append(score)

        new_generation = [new_input_weight, new_input_bias,
                          new_hidden_weight, new_output_weight]

        return new_generation, new_fitness_list

    def show_fitness_graph(self):
        plt.plot(self.fitness_list)
        plt.title("Fitness Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.grid()
        plt.show()

    def evolve(self):
        for gen in range(self.number_of_generation):
            print(f"Generation {gen+1}/{self.number_of_generation}")
            new_gen, new_fit = self.next_generation()
            self.current_generation = new_gen
            self.current_fitness = new_fit
            best_gen_idx = np.argmax(new_fit)
            max_fit = new_fit[best_gen_idx]

            if max_fit > self.best_fitness:
                self.best_fitness = max_fit
                self.best_gen = [
                    self.current_generation[0][best_gen_idx],
                    self.current_generation[1][best_gen_idx],
                    self.current_generation[2][best_gen_idx],
                    self.current_generation[3][best_gen_idx],
                ]
            self.fitness_list.append(max_fit)
        self.show_fitness_graph()
        return self.best_gen, self.best_fitness


def trainer():
    pop_size = 15
    num_of_generation = 100
    learner = NeuralNet(
        env.observation_space.shape[0], 2, env.action_space.n, pop_size)
    init_weight_list, init_fitness_list = learner.run_test()
    optimizer = GA(init_weight_list, init_fitness_list, num_of_generation, pop_size, learner)
    best_weights, best_fitness = optimizer.evolve()
    return best_weights


def test_run_env(params):
    input_w, input_b, hidden_w, out_w = params
    obs, _ = env.reset()
    score = 0
    learner = NeuralNet(
        env.observation_space.shape[0], 2, env.action_space.n, 15)
    for t in range(5000):
        env.render()
        action = learner.forward_prop(obs, input_w, input_b, hidden_w, out_w)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        print(f"time: {t}, fitness: {score}")
        if terminated or truncated:
            break
    print(f"Final score: {score}")


def main():
    params = trainer()
    test_run_env(params)


if(__name__ == "__main__"):
    main()