module gp

using hdfcheb
using TimerOutputs
using LinearAlgebra
using Clustering
using Random
using Statistics
using Plots
using StatsBase


function conj_grad(F, approx_inv, b::AbstractVector;
                   max_iter::Int = 512, tol::Real = 1e-3, precond::Bool = true, verbose::Bool = true)
    conj_grad!(zero(b), F,approx_inv, b, max_iter = max_iter, tol = tol, precond = precond, verbose = verbose)
end

function conj_grad!(x::AbstractVector, F, approx_inv, b::AbstractVector;
                    max_iter::Int = 512, tol::Real = 1e-3, precond::Bool = true, verbose::Bool = true)
    Ax = all(==(0), b) ? zero(x) : F * x
    r = b - Ax
	z = approx_inv * r
	# z = approx_inv(F, r)
	# z = F \ r
    p = copy(z)
    Ap = zero(p)
    rsold = dot(r, z)
	best_res = 100
	best_vec = copy(x)
    for i in 1:max_iter
		Ap = F*p
        # mul!(Ap, F, p)

        alpha = rsold / dot(p, Ap)
        @. x = x + alpha * p
        @. r = r - alpha * Ap

        # if precond # what else changes?
		z = approx_inv * r
		# z = F \ r
        # approx_inv!(z, F, r)
        # end
        rsnew = dot(r, z)
		if rsnew < best_res
			best_res = rsnew
			best_vec = copy(x)
		end
        verbose && println(i, " res ", rsnew)
        sqrt(rsnew) > tol || break
        @. p = z + (rsnew / rsold) * p;
        rsold = rsnew
    end
    return best_vec
end

function m2vov(m)
	return [m[i, :] for i in 1:size(m, 1)]
end

function run_gp(cfg, hdf_relerr_output,nys_relerr_output,hdf_mse_output,nys_mse_output,mse_output)
	radius     = 3*cfg.sigma
	hdf_rtol   = 1e-3

	train_size = convert(Int64, floor(cfg.num_points*0.66666))
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

	@timeit to "assignment" begin
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
	hdf_rel_errs = []
	nys_rel_errs = []
	for i in 1:cfg.num_clusters
		if max_dist_to_ctr[i] < radius
			coverage += cluster_pts[i]
		end
	end

	println("Done assigning points to clusters.")

	println(cluster_pts)
	println(max_dist_to_ctr)
	println(100coverage/train_size, "% coverage")
	end  #timeit
	train_x_vov = m2vov(train_x)
	hdf_rel_errs = []
	nys_rel_errs = []
	covmat = nothing

	println("Populating covariance matrix...")

	@timeit to "covmat pop" covmat = cfg.lambda*diagm(ones(size(train_x,1))) + matkern.(train_x_vov, permutedims(train_x_vov))

	println("Done populating covariance matrix. Populating approx inv...")

	approx_inv = zeros(size(covmat))
	@timeit to "approx inv pop" begin
	for i in 1:cfg.num_clusters
		println("inv ",i)
		pts = [train_x[j,:] for j in cluster_to_pts[i]]
		pt_indices = cluster_to_pts[i]
		approx_inv[pt_indices, pt_indices] = inv(covmat[pt_indices, pt_indices])#TODO(lu)
	end
	end  #timeit

	println("Done populating approx inv. Populating newcovmat...")

	@timeit to "newcovmatpop" newcovmat = matkern.(m2vov(test_x), permutedims(train_x_vov))

	println("Done populating newcovmat. ")

	conditional = conj_grad(covmat, approx_inv, train_y)
	marginal = newcovmat * conditional
	mse = dot(marginal-test_y,marginal-test_y)/length(test_y)
	clust_to_rank = Dict()
	for i in 1:cfg.num_clusters
		println("hdf ",i)
		pts = [train_x[j,:] for j in cluster_to_pts[i]]
		pt_indices = cluster_to_pts[i]
		if max_dist_to_ctr[i] < radius
			truth_mat = matkern.(pts, permutedims(pts))
			U_mat,diag = degen_kern_harmonic(cfg.kern, pts, hdf_rtol,to)
			clust_to_rank[i] = min(size(U_mat, 1), size(U_mat, 2))
			println("Rank ", clust_to_rank[i])
			V_mat = transpose(U_mat)
			guess = U_mat*diagm(diag)*V_mat
			push!(hdf_rel_errs, norm(guess-truth_mat,2)/norm(truth_mat,2))

			rank = min(size(U_mat, 1), size(U_mat, 2))
			covmat[pt_indices,pt_indices] = guess + cfg.lambda*diagm(ones(length(pts)))
		end
	end

	conditional = conj_grad(covmat, approx_inv, train_y)
	marginal = newcovmat * conditional
	hdfmse = dot(marginal-test_y,marginal-test_y)/length(test_y)

	for i in 1:cfg.num_clusters
		println("nys ",i)
		pts = [train_x[j,:] for j in cluster_to_pts[i]]
		pt_indices = cluster_to_pts[i]
		if max_dist_to_ctr[i] < radius
			truth_mat = matkern.(pts, permutedims(pts))
			rank = clust_to_rank[i]


			evecs = eigvecs(truth_mat)
		    sub_evecs = copy(evecs[:,1:rank])
		    lev_scores = [norm(sub_evecs[i, :]) for i in 1:size(sub_evecs,1)]
		    q_set = sample(collect(1:size(truth_mat,1)), Weights(lev_scores), rank, replace=false)


			# q_set = randperm(length(pts))[1:rank]
			Nq =  matkern.(pts, permutedims(pts[q_set]))
			qmat = lu( matkern.(pts[q_set], permutedims(pts[q_set])))
			nystrom_guess = Nq * (qmat \ transpose(Nq))
			push!(nys_rel_errs, norm(nystrom_guess-truth_mat,2)/norm(truth_mat,2))
			covmat[pt_indices,pt_indices] = nystrom_guess +  cfg.lambda*diagm(ones(length(pts)))
		end
	end
	conditional = conj_grad(covmat, approx_inv, train_y)
	marginal = newcovmat * conditional
	nystrommse = dot(marginal-test_y,marginal-test_y)/length(test_y)

	for i in 1:length(hdf_rel_errs)
		write(hdf_relerr_output, string(hdf_rel_errs[i],"\n"))
	end
	for i in 1:length(nys_rel_errs)
		write(nys_relerr_output, string(nys_rel_errs[i],"\n"))
	end
	write(mse_output,string(mse, "\n"))
	write(hdf_mse_output,string(hdfmse, "\n"))
	write(nys_mse_output,string(nystrommse, "\n"))
	display(to)
end

###############################################################################
###############################################################################
###############################################################################

function super_run_gp(cfg)
	hdf_relerr_output = open(string("pyplots/data/gp_hdf_relerr_", cfg.dataset_name,"_",cfg.kernel_name, "_output.txt"), "w")
	nys_relerr_output = open(string("pyplots/data/gp_nys_relerr_", cfg.dataset_name,"_",cfg.kernel_name, "_output.txt"), "w")
	hdf_mse_output    = open(string("pyplots/data/gp_hdf_mse_", cfg.dataset_name,"_",cfg.kernel_name, "_output.txt"), "w")
	nys_mse_output    = open(string("pyplots/data/gp_nys_mse_", cfg.dataset_name, "_",cfg.kernel_name,"_output.txt"), "w")
	mse_output    = open(string("pyplots/data/gp_mse_", cfg.dataset_name, "_",cfg.kernel_name,"_output.txt"), "w")
	# for trial in 1:5
	for trial in 1:5
		run_gp(cfg, hdf_relerr_output,nys_relerr_output,hdf_mse_output,nys_mse_output,mse_output)
	end
	close(hdf_relerr_output)
	close(nys_relerr_output)
	close(hdf_mse_output)
	close(nys_mse_output)
	close(mse_output)
end

###############################################################################
###############################################################################
###############################################################################

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
	sigma_ = 1.5
	if dataset_name == "energy"
		sigma_ = 1.5
	elseif dataset_name == "grid"
		sigma_=1
	elseif dataset_name == "bike"
		sigma_=1
	elseif dataset_name == "power"
		sigma_=1
	end

	num_clusters_ = 30
	if dataset_name=="energy"
		num_clusters_=8  #was 6
	elseif dataset_name == "grid"
		num_clusters_=8
	elseif dataset_name == "bike"
		num_clusters_=8
	elseif dataset_name == "power"
		num_clusters_=8
	end

	file_name_ = "cadata.txt"
	if dataset_name == "energy"
		file_name_ = "energydata_complete.txt"
	elseif dataset_name == "grid"
		file_name_ = "grid_data.txt"
	elseif dataset_name == "bike"
		file_name_ = "bike.txt"
	elseif dataset_name == "power"
		file_name_ = "power.txt"

	end

	num_points_ = 20640
	if dataset_name == "energy"
		num_points_ = 19735
	elseif dataset_name == "grid"
		num_points_ = 10000
	elseif dataset_name == "bike"
		num_points_ = 8760
	elseif dataset_name == "power"
		num_points_ = 9568

	end

	dimension_ = 9
	if dataset_name == "energy"
		dimension_ = 28
	elseif dataset_name == "grid"
		dimension_ = 13
	elseif dataset_name == "bike"
		dimension_ = 10
	elseif dataset_name == "power"
		dimension_=5
	end

	lambda_ = 2.0^(-1.0)

	cauchy_kern(r) = 1/(1+((r^2)/(sigma_^2)))
	gaussian_kern(r) = exp(-0.5(r/sigma_)^2)
	matern15_kern(r) = (1+sqrt(3)*abs(r/sigma_))exp(-sqrt(3)*abs(r/sigma_))
	matern25_kern(r) = (1+sqrt(5)*abs(r/sigma_)+(5/3)*(r/sigma_)^2)exp(-sqrt(5)*abs(r/sigma_))

	if kernel_name == "cauchy"
		return exp_config(kernel_name, dataset_name,file_name_, sigma_, num_clusters_,
			num_points_, dimension_, lambda_, cauchy_kern)
	elseif kernel_name == "gaussian"
		return exp_config(kernel_name, dataset_name, file_name_, sigma_, num_clusters_,
			num_points_, dimension_, lambda_, gaussian_kern)
	elseif kernel_name == "matern15"
		return exp_config(kernel_name, dataset_name, file_name_, sigma_, num_clusters_,
			num_points_, dimension_, lambda_, matern15_kern)
	elseif kernel_name == "matern25"
		return exp_config(kernel_name, dataset_name, file_name_, sigma_, num_clusters_,
			num_points_, dimension_, lambda_, matern25_kern)
	end

end


dataset_name="energy"
kernel_name = "cauchy"
cfg = get_config(kernel_name, dataset_name)
super_run_gp(cfg)



end
