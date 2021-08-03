
import torch
import random, copy
import networkx as nx
import dgl
import cplex
import dgl.function as fn
import numpy as np
def new_calculate_cut(state, graph):
    with graph.local_scope():
        graph.ndata['assignment'] = state
        graph.apply_edges(lambda edges: {'cut' : abs(edges.src['assignment'] - edges.dst['assignment'])})
        graph.update_all(fn.copy_e('cut', 'x'), fn.sum('x', 'h'))

        # assert(torch.sum(graph.ndata['h']) == self.calculate_cut(state, graph))
        cuts = dgl.readout_nodes(graph, 'h', op = 'sum')
        return cuts
def KargerAlgrotihm(g, k, m):




    # k = k + num_cc - 1
    # print(k)
    def contraction(g, k):
        g = copy.deepcopy(g)

        with g.local_scope():
            g.ndata['h'] = torch.tensor([i for i in range(g.num_nodes())])
            cluster = [[i] for i in range(g.num_nodes())]
            while g.num_nodes() > k  :


                # if g.num_nodes() == 3:
                #     exit()

                # print(g)
                # print(g)
                # print("g.ndata['h'] = ", g.ndata['h'])

                e1, e2 = g.edges()
                # print("\nGraph: ")
                # print(e1)
                # print(e2)


                rand_edge_id = random.randint(0, g.num_edges()-1)



                edge = g.find_edges(torch.tensor([rand_edge_id]))
                # print(edge)

                u, v = edge

                if (u == v):
                    continue


                cluster[g.ndata['h'][v]] += cluster[g.ndata['h'][u]]


                ## u to be removed
                

                # print(g.out_edges(u))

                tmp = g.out_edges(u)[1]

                qaq = v.repeat(tmp.size())


                # print(qaq)
                # print(tmp)

                g.add_edges(qaq, tmp)

                g.remove_nodes(u)
                g = dgl.to_simple(g)
                g = dgl.to_bidirected(g, copy_ndata  = True)
                g = dgl.remove_self_loop(g)
                # print("g.ndata['h'] = ", g.ndata['h'])
                # print(cluster)


            # print("g.ndata['h'] = ", g.ndata['h'])
            final_cluster = []
            for i in range(k):
                final_cluster.append(cluster[g.ndata['h'][i]])
            # print(final_cluster)

            # print()
            return g.num_nodes(), final_cluster


    min_value = g.num_edges() + 1
    min_cluster = []
    for i in range(m):
        value, cluster = contraction(g, k)

        if min_value > value:
            min_value = value 
            min_cluster = cluster


    # print("min_value = ", min_value)
    # print("min_cluster", min_cluster)

    return min_value, min_cluster

def graphPartition(g, k, m = 0):
    g = copy.deepcopy(g)
    g = g.to("cpu")
    if not m:
        m = 10

    value, cluster = KargerAlgrotihm(g, k, m)
    return cluster
    all_id = torch.arange(0, g.num_edges())

    mask = torch.tensor([True for _ in range(g.num_edges())] )

    in_cluster_eid = torch.tensor([]).to(torch.int64)
    for i in range(k):
        sg = dgl.node_subgraph(g, cluster[i])
        tmp = sg.edata[dgl.EID]  # original edge IDs
        in_cluster_eid = torch.cat((in_cluster_eid, tmp))
        

    # print(in_cluster_eid)
    mask[in_cluster_eid] = False

    out_cluster_eid = all_id[mask]
    # print(out_cluster_eid)

    return in_cluster_eid, out_cluster_eid

def max_cut_solve(graphs):

    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    prob.set_problem_name("MAX Cut")
    prob.set_problem_type(cplex.Cplex.problem_type.LP)
    prob.objective.set_sense(prob.objective.sense.maximize)
    # names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    n = graphs.number_of_nodes()
    edges = [(i, j) if i > j else (j, i) for i, j in zip(graphs.edges()[0].numpy(), graphs.edges()[1].numpy())]


    names = []
    w_obj = []
    # edge ij, i > j
    for i in range(n):
        for j in range(i):
            names.append(str(i)+str(j))
            w_obj.append(int((i, j) in edges))



    low_bnd = [0 for x in range(n*(n-1)//2)]
    upr_bnd = [1 for x in range(n*(n-1)//2)]

    prob.variables.add(names=names, obj=w_obj, lb=low_bnd, ub=upr_bnd)
    all_int = [(var, prob.variables.type.integer) for var in names]
    prob.variables.set_types(all_int)
    constraints = []
    rhs = []
    for i in range(n):
        for j in range(i):
            for k in range(j):
                constraints.append([[str(i)+str(j), str(i)+str(k), str(j)+str(k)], [-1, 1, 1]])
                rhs.append(0)
                constraints.append([[str(i)+str(j), str(i)+str(k), str(j)+str(k)], [-1, -1, -1]])
                rhs.append(-2)


    # for src, dist in zip(graphs.edges()[0].numpy(), graphs.edges()[1].numpy()):
    #     constraints.append([[str(src), str(dist)], [1, 1]])

    constraint_names = ["".join(x[0]) for x in constraints]
    # rhs = [1] * len(constraints)
    constraint_senses = ["G"] * len(constraints)
    prob.linear_constraints.add(names=constraint_names,
                                lin_expr=constraints,
                                senses=constraint_senses,
                                rhs=rhs)
    prob.solve()


    # print(prob.solution.get_values())
    # print(prob.solution.get_values())
    # print(prob.solution.get_values())
    # print(prob.solution)
    cut_edges = [i and j for i, j in zip(prob.solution.get_values(), w_obj)]
    cut_u = []
    cut_v = []
    cnt = 0
    for i in range(n):
        for j in range(i):
            if cut_edges[cnt]:
                cut_u.append(i)
                cut_u.append(j)
                cut_v.append(j)
                cut_v.append(i)
            cnt += 1


    bg = dgl.to_bidirected(graphs)
    tmp = bg.edge_ids(torch.tensor(cut_u), torch.tensor(cut_v))

    # print(bg)
    bg.remove_edges(tmp)
    # print(bg)
    nx_g = bg.to_networkx().to_undirected()

    assignment = [0 for i in range(n)]
    for cc in max(nx.connected_components(nx_g), key=len):
        assignment[cc] = 1
    assignment.append(sum(cut_edges))
    return assignment

# eco-dqn obs-2
# Need check
# Immediate cut change if vertex state is flipped.
def delta_cut(state, graph):
    with graph.local_scope():
        e1, e2 = graph.edges()
        # print(e1)
        # print(e2)
        # print(state)
        graph.ndata['x'] = state
        graph.apply_nodes(lambda nodes: {'h' : nodes.data['x'] * 2 - 1})
        # print(graph.ndata['h'])
        graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'n'))
        # print(graph.ndata['n'])
        graph.apply_nodes(lambda nodes: {'delta' : nodes.data['h'] * nodes.data['n']})
        # print(graph.ndata['delta'])
        # naive_result = naive_cal_cut(state, graph)
        # print("graph.edata['cut'] = ", graph.edata['cut'])
        # print("torch.sum(graph.edata['cut']) = ", torch.sum(graph.edata['cut']))
        # print("naive_cal_cut(state, graph)== ", naive_result)

        # assert(torch.sum(graph.edata['cut']) == naive_result)
        # exit()
        return graph.ndata['delta']  


def number_of_inc_cut(delta_cut_value):

    zero = torch.zeros_like(delta_cut_value)
    one = torch.ones_like(delta_cut_value)
    x = torch.where(delta_cut_value > 0, one, delta_cut_value)
    x = torch.where(x < 0, zero, x)
    return torch.sum(x)

def mvc_optimal_solve(graphs):

    prob = cplex.Cplex()
    prob.set_log_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)
    prob.set_results_stream(None)
    prob.set_problem_name("Minimum Vertex Cover")
    prob.set_problem_type(cplex.Cplex.problem_type.LP)
    prob.objective.set_sense(prob.objective.sense.minimize)
    # names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    names = [str(x) for x in range(graphs.number_of_nodes())]
    w_obj = [1 for x in range(graphs.number_of_nodes())]
    low_bnd = [0 for x in range(graphs.number_of_nodes())]
    upr_bnd = [1 for x in range(graphs.number_of_nodes())]
    prob.variables.add(names=names, obj=w_obj, lb=low_bnd, ub=upr_bnd)
    all_int = [(var, prob.variables.type.integer) for var in names]
    prob.variables.set_types(all_int)
    constraints = []

    for src, dist in zip(graphs.edges()[0].numpy(), graphs.edges()[1].numpy()):
        constraints.append([[str(src), str(dist)], [1, 1]])

    constraint_names = ["".join(x[0]) for x in constraints]
    rhs = [1] * len(constraints)
    constraint_senses = ["G"] * len(constraints)
    prob.linear_constraints.add(names=constraint_names,
                                lin_expr=constraints,
                                senses=constraint_senses,
                                rhs=rhs)
    prob.solve()


    # print(prob.solution.get_values())
    # print(prob.solution.get_values())
    # print(prob.solution.get_values())


    return prob.solution.get_values()




def max_cut_greedy(graph):
    state = torch.zeros(graph.num_nodes())
    while 1:
        tmp = delta_cut(state, graph)



        if (number_of_inc_cut(tmp) == 0):
            total = new_calculate_cut(state, graph).item()
            ass = state.tolist()
            ass.append(total/2) 
            return ass

        index = tmp.argmax()
        state[index] = 1 - state[index]




def opt_sol(dataset):

    cplex_sol = []
    greedy_sol = []
    for idx, (g, _) in enumerate(dataset):
        # print(idx)
        cplex_sol.append(max_cut_solve(g)[-1])
        greedy_sol.append(max_cut_greedy(g)[-1])

    return {"cplex_sol": cplex_sol, "greedy_sol": greedy_sol}


