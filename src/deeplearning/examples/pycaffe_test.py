import argparse
import numpy as np
import time
import caffe
import resource
from caffe import layers as L, params as P
import lmdb
from shutil import copyfile

parser = argparse.ArgumentParser()

parser.add_argument('--full_layers', type=int, default=None, nargs='+',
                    help="fully connected layers separated by spaces")
parser.add_argument('--conv_layers', type=int, default=None, nargs='+',
                    help="convolution layer features separated by spaces")
parser.add_argument('--epochs', type=int, default=30, help='epochs')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--train_batch', type=int, default=10, help='batch size for training')
parser.add_argument('--test_batch', type=int, default=100, help='batch size for testing')
parser.add_argument('--time', type=float, default=-1, help='time to run')
args = parser.parse_args()


def build_network(batch_size, lmdb_name):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb_name,
                             transform_param=dict(scale=1. / 255), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.relu1 = L.ReLU(n.pool1, in_place=True)
    n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
    n.loss = L.SoftmaxWithLoss(n.score, n.label)
    """
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
    n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.reluc1 = L.ReLU(n.pool1, in_place=True)
    n.conv2 = L.Convolution(n.reluc1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
    n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
    n.reluc2 = L.ReLU(n.pool2, in_place=True)
    n.fc1 = L.InnerProduct(n.reluc2, num_output=500, weight_filler=dict(type='xavier'))
    n.fc2 = L.InnerProduct(n.data, num_output=500, weight_filler=dict(type='xavier'))
    n.relu2 = L.ReLU(n.fc2, in_place=True)
    n.fc1 = L.InnerProduct(n.relu2, num_output=500, weight_filler=dict(type='xavier'))
    """
    return n.to_proto()


def write_network(args_in):
    with open('train.prototxt', 'w') as f:
        f.write(str(build_network(args_in.train_batch, 'mnist_train_lmdb')))
    with open('test.prototxt', 'w') as f:
        f.write(str(build_network(args_in.test_batch, 'mnist_test_lmdb')))


def write_solver(args_in, training_samples, testing_samples):
    training_batches_per_epoch_int = training_samples / args_in.train_batch
    testing_batches_per_epoch_int = testing_samples / args_in.test_batch

    total_number_of_batches = training_batches_per_epoch_int * args_in.epochs
    with open('solver.prototxt', 'w') as f:
        f.write('train_net: "train.prototxt"\n')
        f.write('test_net: "test.prototxt"\n')
        f.write('test_iter: %i\n' % 0)
        f.write('test_interval: %i\n' % total_number_of_batches)
        f.write('base_lr: {}\n'.format(args_in.learning_rate))
        f.write('momentum: 0.9\n')
        f.write('weight_decay: 0.0005\n')
        f.write('lr_policy: "inv"\n')
        f.write('gamma: 0.0001\n')
        f.write('power: 0.75\n')
        f.write('display: %i\n' % 0)
        f.write('max_iter: %i\n' % total_number_of_batches)
        f.write('snapshot: %i\n' % 0)
        f.write('snapshot_prefix: "network"\n')
        f.write('solver_mode: CPU')
    return training_batches_per_epoch_int, testing_batches_per_epoch_int

start = time.time()
write_network(args)

training_batches_per_epoch, testing_batches_per_epoch = write_solver(args, 60000, 10000)  # make sample numbers dynamic

test_acc = np.zeros(int(np.ceil(args.epochs)))

caffe.set_mode_cpu()
solver = caffe.SGDSolver('solver.prototxt')

print "Startup:", time.time()-start

s = time.time()
for epoch in range(args.epochs):
    estart = time.time()
    solver.step(60000/args.train_batch)
    correct = 0
    for test_it in range(100):
        solver.test_nets[0].forward()
        correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['label'].data)
    test_acc[epoch] = float(correct)/10000
    #temp = solver.net.params['fc1'][0].data[0]
    #weight_sum = np.sum(np.square(temp))
    print epoch+1, test_acc[epoch], time.time()-estart, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0 # weight_sum
    # solver.net.save('/home/charles/Desktop/network.txt')

print "Best Accuracy:", max(test_acc)
print "Average time:", (time.time()-s)/args.epochs

"""
        old_out = low_out
        sum_error = sum(error_list)
        sum_error_list = comm.allreduce(sum_error)
        classification_list, low_out, number_in, reset = se.elimination(classification_list, error_list, low_out, number_of_samples, old_out, global_number_of_samples, sum_error_list, comm, elim_rate, max_eon, epoch_in_eon, fixed_eon, pct_elim)
        reset = comm.bcast(reset) # root==0, so root result is broadcast everywhere
        sum_number_in = comm.allreduce(number_in)
        epoch_in_eon += 1
        if reset == True:
            eff_dat = full_dat
            eff_lab = full_lab
            classification_list = ['in'] * number_of_samples
            reset = False
            epoch_in_eon = 0
        accurate = sess.run(accuracy, feed_dict={x: test_dat, y_: test_lab})
        if 0 == rank:
            print str(epoch+1) + "," + str(time.time()-t) + "," + str(accurate) + "," + str(low_out) + "," + str(sum_number_in) + "," + str(global_number_of_samples-sum_number_in) + "," + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000.0) + "," + str(sum_error_list)
        truth = [inde for inde in range(len(classification_list)) if classification_list[inde] != 'out']
        eff_dat = eff_dat[truth,]
        eff_lab = eff_lab[truth,]
        classification_list = [a for a in classification_list if a != 'out']
"""

"""
def elimination(classification_list, error_list, low_out_old, number_of_samples, old_out, global_number_of_samples, global_error, comm, elrate, max_eon, epoch_in_eon, fixed_eon, pct_elim):
    temp_errors = [error_list[i] for i in range(len(error_list)) if classification_list[i] != 'out']
    reset = False
    if pct_elim == None and temp_errors != []:
        bottom = min(temp_errors)
        middle = sum(temp_errors)/float(len(temp_errors))
        if fixed_eon == True:
            low_out = middle - (middle-bottom)/elrate
        else:
            low_out = min([low_out_old, middle - (middle-bottom)/elrate])
        for i in range(len(classification_list)):
            if classification_list[i] != 'out':
                if error_list[i] <= low_out:
                    classification_list[i] = 'out'
                else:
                    classification_list[i] = 'in'
        number_in = classification_list.count('in')
    elif pct_elim != None:
        index = int(len(temp_errors) * pct_elim)
        dupe = temp_errors
        dupe.sort()
        if dupe != []:
            cutoff = dupe[index]
            for i in range(len(classification_list)):
                if error_list[i] <= cutoff:
                    classification_list[i] = 'out'
                    else:
                    classification_list[i] = 'in'
        else:
            reset = True
        low_out = 0
        number_in = classification_list.count('in')
    else:
        low_out = 0
    if epoch_in_eon >= max_eon:
        reset = True
    if temp_errors == []:
        reset = True
    if global_error <= global_number_of_samples * old_out and fixed_eon == False:
            reset = True
        reset = comm.bcast(reset)
        if reset:
            classification_list = ['in'] * number_of_samples
            number_in = len(classification_list)
            reset = True
    return classification_list, low_out, number_in, reset
"""