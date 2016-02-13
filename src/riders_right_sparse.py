__author__ = 'pratik'

import nimfa
import numpy as np
import argparse
import mdl


if __name__ == "__main__":
    np.random.seed(1000)
    argument_parser = argparse.ArgumentParser(prog='compute right sparsity')
    argument_parser.add_argument('-nf', '--node-feature', help='node-feature matrix file', required=True)
    argument_parser.add_argument('-o', '--output-prefix', help='output prefix', required=True)
    argument_parser.add_argument('-od', '--output-dir', help='output dir', required=True)

    args = argument_parser.parse_args()

    node_feature = args.node_feature
    out_prefix = args.output_prefix
    out_dir = args.output_dir

    refex_features = np.loadtxt(node_feature, delimiter=',')
    np.savetxt(out_dir + '/out-' + out_prefix + '-ids.txt', X=refex_features[:, 0])
    actual_fx_matrix = refex_features[:, 1:]

    n, f = actual_fx_matrix.shape
    print 'Number of Features: ', f
    print 'Number of Nodes: ', n

    number_bins = int(np.log2(n))
    max_roles = min([n, f])
    best_G = None
    best_F = None

    mdlo = mdl.MDL(number_bins)
    minimum_description_length = 1e20
    min_des_not_changed_counter = 0
    sparsity_threshold = 1.0
    for rank in xrange(1, max_roles + 1):
        snmf = nimfa.Snmf(actual_fx_matrix, seed="random_vcol", version='r', rank=rank, beta=2.0)
        snmf_fit = snmf()
        G = np.asarray(snmf_fit.basis())
        F = np.asarray(snmf_fit.coef())

        code_length_G = mdlo.get_huffman_code_length(G)
        code_length_F = mdlo.get_huffman_code_length(F)

        model_cost = code_length_G * (G.shape[0] + G.shape[1]) + code_length_F * (F.shape[0] + F.shape[1])
        estimated_matrix = np.asarray(np.dot(G, F))
        loglikelihood = mdlo.get_log_likelihood(actual_fx_matrix, estimated_matrix)
        err = snmf_fit.distance(metric='kl')

        description_length = model_cost + err  #- loglikelihood

        if description_length < minimum_description_length:
            minimum_description_length = description_length
            best_G = np.copy(G)
            best_F = np.copy(F)
            min_des_not_changed_counter = 0
        else:
            min_des_not_changed_counter += 1
            if min_des_not_changed_counter == 5:
                break
        try:
            print 'Number of Roles: %s, Model Cost: %.2f, -loglikelihood: %.2f, ' \
                  'Description Length: %.2f, MDL: %.2f (%s)' \
                  % (rank, model_cost, loglikelihood, description_length, minimum_description_length, best_G.shape[1])
        except Exception:
            continue


    print 'MDL has not changed for these many iters:', min_des_not_changed_counter
    print '\nMDL: %.2f, Roles: %s' % (minimum_description_length, best_G.shape[1])
    np.savetxt(out_dir + '/' + 'out-' + out_prefix + "-nodeRoles.txt", X=best_G)
    np.savetxt(out_dir + '/' + 'out-' + out_prefix + "-roleFeatures.txt", X=best_F)
