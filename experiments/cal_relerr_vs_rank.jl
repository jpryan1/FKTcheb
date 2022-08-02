module gp

using hdfcheb
using TimerOutputs
using LinearAlgebra
using Clustering
using Random
using Statistics
using StatsBase

function m2vov(m)
	return [m[i, :] for i in 1:size(m, 1)]
end

function run_gp(cfg)
	radius     = 3*cfg.sigma

	train_size = convert(Int64, floor(cfg.num_points*0.666666))
	test_size  = cfg.num_points-train_size
	matkern(x,y) = cfg.kern(norm(x-y))

	println("Reading data...")

	X = zeros(cfg.num_points, cfg.dimension)
	open(cfg.file_name) do f
		line = 1
		while ! eof(f)
			# read a new / next line for every iteration
			splitted = split(readline(f),",")
			if cfg.dataset_name=="energy"
				splitted = splitted[2:end]
			end
			if cfg.dataset_name=="bike"
				splitted = splitted[2:(end-3)]
			end
			if cfg.dataset_name=="cadata"
				# txt file has spurious comma at beginning
				splitted = splitted[2:end]
				endsplit = split(splitted[end], " ")
				splitted[end] = endsplit[1]
				push!(splitted, endsplit[2])
			end
			s = [parse(Float64, x) for x in splitted]
			# println(size(X[line,:]))
			# println(size(s))
			X[line, :] .= s
			line += 1
		end
	end

	println("Done reading data. Normalizing data...")
	good_indices = []
	for i in 1:size(X, 1)
		if norm(X[i,2:end])>0
			push!(good_indices, i)
		else
			println("FOUND BAD ",i)
		end
	end
	X = X[randperm(cfg.num_points), :]
	train_x = X[1:train_size, :]
	test_x  = X[(train_size+1):(train_size+test_size), :]
	mins = zeros(cfg.dimension)
	maxs  = zeros(cfg.dimension)
	for i in 1:(cfg.dimension)
		mins[i] = minimum(train_x[:, i])
		maxs[i]  = maximum( train_x[:,i])
		if maxs[i]-mins[i]==0
				train_x[:, i] .= 0
				test_x[:, i] .= 0
		else
			train_x[:, i] .= (train_x[:, i] .- mins[i])/(maxs[i]-mins[i])
			test_x[:, i] .= (test_x[:, i] .- mins[i])/(maxs[i]-mins[i])
		end
	end

	labels = 1
	features = collect(2:size(train_x,2))
	if cfg.dataset_name=="power" || cfg.dataset_name=="grid"
		labels = size(train_x,2)
		features = collect(1:size(train_x,2)-1)
	end
	train_y = train_x[:, labels]
	train_x = train_x[:, features]
	test_y  = test_x[:, labels]
	test_x  = test_x[:, features]
	println("Done normalizing data. Finding cluster centers (kmeans)...")

	to = TimerOutput()
	@timeit to "kmeans" R = kmeans(transpose(train_x), cfg.num_clusters)

	println("Done finding cluster centers. Assigning points to clusters...")

	# @timeit to "assignment" begin
	max_dist_to_ctr = zeros(cfg.num_clusters)
	cluster_pts = zeros(cfg.num_clusters)
	cluster_to_pts = [[] for i in 1:cfg.num_clusters]
	assigns = assignments(R)
	for i in 1:train_size
		assignment = assigns[i]
		cluster_pts[assignment] += 1
		push!(cluster_to_pts[assignment], i)
		ctr = R.centers[:,assignment]
		pt = train_x[i, :]
		dist = norm(pt-ctr)
		max_dist_to_ctr[assignment] = max(max_dist_to_ctr[assignment], dist)
	end
	coverage = 0
	println("CLUSTER POINTS ", cluster_pts)
	hdf_rel_errs = []
	nys_rel_errs = []
	for i in 1:cfg.num_clusters
		if max_dist_to_ctr[i] < radius
			coverage += cluster_pts[i]
		end
	end

	println("Done assigning points to clusters.")
	clust_to_rank = Dict()

	rank_stride = 5
	# for i in 1:cfg.num_clusters
		for i in [1]
		# println("hdf ",i)
		pts = [train_x[j,:] for j in cluster_to_pts[i]]
		truth_mat = matkern.(pts, permutedims(pts))
		print(length(pts), ",")
		pt_indices = cluster_to_pts[i]
		if max_dist_to_ctr[i] < radius

			hdf_ranks = []
			hdf_errs = []
			for rtol_pow in -5:-1:-18
			    # println(rtol_pow)
			    rtol = 2.0^rtol_pow
			    U_mat, diag_mat = degen_kern_harmonic(cfg.kern, pts, rtol,to)
			    V_mat = transpose(U_mat)
			    hdf_rank = size(V_mat, 1)
				if hdf_rank > length(pts)
				    break
				end
			    hdf_guess = U_mat*diagm(diag_mat)*V_mat
			    if length(hdf_ranks) > 0 && hdf_rank > hdf_ranks[end]+rank_stride
			        last_rank = hdf_ranks[end]
			        starting_point = hdf_rank
			        while starting_point > last_rank+rank_stride
			            starting_point -= rank_stride
			        end
			        for new_rank in starting_point:rank_stride:(hdf_rank-rank_stride)
			            push!(hdf_ranks, new_rank)
			            new_hdf_guess = U_mat[:,1:new_rank]*diagm(diag_mat[1:new_rank])*V_mat[1:new_rank,:]
			            push!(hdf_errs, norm(new_hdf_guess-truth_mat,2)/norm(truth_mat,2))
			            # println(hdf_ranks[end], " ", hdf_errs[end])
			        end
			    end
			    push!(hdf_ranks, hdf_rank)
			    push!(hdf_errs, norm(hdf_guess-truth_mat,2)/norm(truth_mat,2))
			end
			new_hdf_ranks = []
			new_hdf_errs = []
			for i in length(hdf_ranks):-1:1
				rank = hdf_ranks[i]
				if length(new_hdf_ranks) > 0
					if rank == new_hdf_ranks[end]
						continue
					end
				end
				push!(new_hdf_ranks, rank)
				push!(new_hdf_errs, hdf_errs[i])
			end
			nys_errs = []
			for i in 1:length(new_hdf_ranks)
				rank=new_hdf_ranks[i]

				evecs = eigvecs(truth_mat)
				sub_evecs = copy(evecs[:,1:rank])
				lev_scores = [norm(sub_evecs[i, :]) for i in 1:size(sub_evecs,1)]
				q_set = sample(collect(1:size(truth_mat,1)), Weights(lev_scores), rank, replace=false)
				# q_set = randperm(length(pts))[1:rank]
				Nq =  matkern.(pts, permutedims(pts[q_set]))
				qmat = lu( matkern.(pts[q_set], permutedims(pts[q_set])))
				nystrom_guess = Nq * (qmat \ transpose(Nq))
				nys_err =norm(nystrom_guess-truth_mat,2)/norm(truth_mat,2)
				push!(nys_errs, nys_err)
			end
			for i in 1:length(new_hdf_ranks)
				print(new_hdf_ranks[i],",",nys_errs[i],",",new_hdf_errs[i],",")
			end
			println("")
			# println(nys_errs)

			# U_mat,diag = degen_kern_harmonic(cfg.kern, pts, hdf_rtol, to)
			# clust_to_rank[i] = min(size(U_mat, 1), size(U_mat, 2))
			# println("Rank ", clust_to_rank[i])
			# V_mat = transpose(U_mat)
			# guess = U_mat*diagm(diag)*V_mat
			# hdf_err = norm(guess-truth_mat,2)/norm(truth_mat,2)
			# rank = min(size(U_mat, 1), size(U_mat, 2))
			# evecs = eigvecs(truth_mat)
			# sub_evecs = copy(evecs[:,1:rank])
			# lev_scores = [norm(sub_evecs[i, :]) for i in 1:size(sub_evecs,1)]
			# q_set = sample(collect(1:size(truth_mat,1)), Weights(lev_scores), rank, replace=false)
			# # q_set = randperm(length(pts))[1:rank]
			# Nq =  matkern.(pts, permutedims(pts[q_set]))
			# qmat = lu( matkern.(pts[q_set], permutedims(pts[q_set])))
			# nystrom_guess = Nq * (qmat \ transpose(Nq))
			# nys_err =norm(nystrom_guess-truth_mat,2)/norm(truth_mat,2)
			# println("Rank: ", rank, " err diff ",nys_err-hdf_err )
		end

	end
end

###############################################################################
###############################################################################
###############################################################################

function super_run_gp(cfg)
	run_gp(cfg)


end

struct exp_config
	kernel_name::String
	dataset_name::String
	file_name::String
	sigma::Float32
	num_clusters::Int64
	num_points::Int64
	dimension::Int64
	lambda::Float32
	kern
end

function get_config(kernel_name, dataset_name)
	sigma_ = 0.5

	num_clusters_ = 8

	file_name_ = "power.txt"


	num_points_ = 9568

	dimension_ = 5

	lambda_ = 2.0^(-1.0)

	cauchy_kern(r) = 1/(1+((r^2)/(sigma_^2)))
	return exp_config(kernel_name, dataset_name,file_name_, sigma_, num_clusters_,
		num_points_, dimension_, lambda_, cauchy_kern)


end


dataset_name="power"
kernel_name = "cauchy"
cfg = get_config(kernel_name, dataset_name)
super_run_gp(cfg)



end
