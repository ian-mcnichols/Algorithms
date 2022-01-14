using Random
using Graphs
using HDF5

function fluid_communities(graph, num_communities, max_iter)
    v_random = rand(1:nv(graph), nv(graph))
    communities = rand(1:1, nv(graph))
    densities = rand(0:0, nv(graph))
    for i in 1:num_communities
        v = v_random[i]
        communities[v] = i
        densities[v] = 1.0
    end
    converged = false
    num_iterations = 0
    while num_iterations < max_iter && converged == false
        converged = true
        for v in v_random
            v_new_community = community_update(graph, v, communities, densities)
            if v_new_community != communities[v]
                v_old_community = communities[v]
                println("communities[v]: ", communities[v], " v_new_community: ", v_new_community)
                communities[v] = v_new_community
                density_update(graph, v_old_community, densities, communities)
                density_update(graph, v_new_community, densities, communities)
                converged = false
            end
        end
    end
end

function density_update(graph, community_to_update, densities, communities)
    vertices_list = getindex(communities, community_to_update)
    v_length = length(vertices_list)
    for v in vertices_list
        densities[v] = 1.0 / v_length
    end
end

function community_update(graph, v, communities, densities)
    community_densities = rand(-1:-1, nv(graph))
    community_densities[v] = densities[v]
    for w in neighbors(graph, v)
        if community_densities[communities[w]] != -1
            communities[communities[w]] += densities[w]
        else
            community_densities[communities[w]] = densities[w]
        end
    end
    community_density_prime = community_densities[v]
    max_density = maximum(community_densities)
    if community_densities[communities[v]] < max_density
        max_inds = getindex(community_densities, max_density)
        community_density_prime = community_densities[rand(max_inds, 1)]
    end
    println("community_density_prime: ", community_density_prime[0])
    return community_density_prime
end

function read_h5_graph(dataset_name, graph_no)
    println("Reading graph number "*graph_no* " from dataset "*dataset_name)
    # Read graph information from h5 file
    h5file = h5open(dataset_name, "r")
    community_sizes = read(h5file["Community Sizes/"*graph_no])
    num_clusters = length(community_sizes)
    num_nodes = length(read(h5file["Degrees/"*graph_no]))
    edges_a = read(h5file["Edges/"*graph_no*"_a"])
    edges_b = read(h5file["Edges/"*graph_no*"_b"])
    close(h5file)

    # Add each edge to the graph
    new_graph = SimpleGraph(num_nodes)
    for i in 1:length(edges_a)
        add_edge!(new_graph, edges_a[i], edges_b[i])
    end
    return new_graph, num_nodes
end


function main()
    dataset = "/mnt/alam01/slow02/imcnichols/FORG/graphs/graphdata4.h5"
    graph = "5"
    graph_obj, n = read_h5_graph(dataset, graph)
    fluid_communities(graph_obj, 4, 100)
    
end

main()