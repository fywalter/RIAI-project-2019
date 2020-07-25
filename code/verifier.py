import argparse
import torch
from networks import FullyConnected, Conv
import os
import time
import numpy as np

main_dir = os.getcwd()
DEVICE = 'cpu'
INPUT_SIZE = 28


def relu_transform(inputs, error_term):
    """
    Relu transformer
    :param inputs: torch.Tensor - size: (1, C, H, W)
    :param error_term: torch.Tensor - size: (error_term_num, C, H, W)
    :return:
    """
    error_apt = torch.sum(torch.abs(error_term), dim=0, keepdim=True).view(inputs.size())
    ux = inputs + error_apt
    lx = inputs - error_apt

    c = error_term.shape[1]
    h = error_term.shape[2]
    w = error_term.shape[3]
    assert c == inputs.shape[1], 'Relu transformer channel num error'
    assert h == inputs.shape[2], 'Relu transformer height num error'
    assert w == inputs.shape[3], 'Relu transformer width num error'

    err_num_ori = error_term.shape[0]
    case_idx = torch.where(((ux[0, :, :, :] > 0) & (lx[0, :, :, :] < 0)))   # where to compute lambda
    point_num = case_idx[0].shape[0]                                        # how many new error term to add
    error_term_new = torch.zeros((err_num_ori + point_num, c, h, w))        # new error term with new size
    outputs = torch.zeros(inputs.size())
    outputs[lx >= 0] = inputs[lx >= 0]                                      # lower bound >=0
    error_term_new[:err_num_ori, :, :, :] = error_term                      # lower bound >=0 error terms stay unchanged
    error_term_new[:, ux[0, :, :, :] <= 0] = 0                              # upper bound <= 0
    ux_select = ux[0][case_idx]
    lx_select = lx[0][case_idx]
    error_term_select = error_term[:, case_idx[0], case_idx[1], case_idx[2]]
    inputs_select = inputs[0][case_idx]
    slopes = ux_select / (ux_select - lx_select)    #lambda
    outputs[0][case_idx] = slopes * inputs_select - slopes * lx_select / 2
    error_term_new[:err_num_ori, case_idx[0], case_idx[1], case_idx[2]] = slopes.view((1, -1)) * error_term_select
    new_error_terms = -slopes * lx_select / 2
    for i in range(point_num):
        c_idx, h_idx, w_idx = case_idx[0][i], case_idx[1][i], case_idx[2][i]
        error_term_new[err_num_ori + i, c_idx, h_idx, w_idx] = new_error_terms[i]
    return outputs, error_term_new


def affine_transform(layer, inputs, error_term):
    """
    Affine transformer
    :param layer: torch.nn.module - layer to computer convex relaxation
    :param inputs:  torch.Tensor - size: (1, 1, feature_length, 1)
    :param error_term:  torch.Tensor - size: (error_term_num, 1, feature_length, 1)
    :return:
    """
    assert inputs.size()[2] == error_term.size()[2], 'Affine transformer error_term dimension error'
    outputs = (layer.weight.mm(inputs.detach()[0, 0, :, :])).view((layer.weight.shape[0])) + layer.bias
    outputs = outputs.view(1, 1, outputs.size()[0], 1)
    error = error_term[:, 0, :, 0].mm(layer.weight.permute(1, 0))  # transpose to do matrix mut
    error = error.view(error.size()[0], 1, error.size()[1], 1)
    return outputs, error


def conv_transform(layer, inputs, error_term):
    """
    Convolution transformer
    :param layer: torch.nn.module - layer to computer convex relaxation
    :param inputs: torch.Tensor - size: (1, C, H, W)
    :param error_term: torch.Tensor - size: (error_term_num, C, H, W)
    :return:
    """
    padding_1, padding_2 = layer.padding[0], layer.padding[1]
    stride_1, stride_2 = layer.stride[0], layer.stride[1]
    kernel_size_1, kernel_size_2 = layer.weight.size()[2], layer.weight.size()[3]
    assert kernel_size_1 == kernel_size_2, 'Convolution kernel sizes in 2 dimension are not equal!'
    assert padding_1 == padding_2, 'padding sizes not equal'
    assert stride_1 == stride_2, 'stride not equal'
    (error_term_num, c, h, w) = error_term.size()
    assert c == inputs.shape[1], 'Conv transformer channel num error'
    assert h == inputs.shape[2], 'Conv transformer height num error'
    assert w == inputs.shape[3], 'Conv transformer width num error'
    assert h == w, 'Conv: h and w not equal'

    outputs = torch.nn.functional.conv2d(inputs, layer.weight, layer.bias, stride=layer.stride,
                                         padding=layer.padding)
    error_term = error_term.view((error_term_num, c, h, w))
    error = torch.nn.functional.conv2d(error_term, layer.weight, stride=layer.stride,
                                       padding=layer.padding)
    return outputs, error


def analyze(net, inputs, eps, true_label):
    start_pred = time.time()
    inputs_lx = inputs.detach() - eps * 1  # lower bound
    inputs_ux = inputs.detach() + eps * 1  # upper bound
    inputs_lx[inputs_lx < 0] = 0
    inputs_ux[inputs_ux > 1] = 1
    inputs = (inputs_ux - inputs_lx) / 2 + inputs_lx
    error_term_apt = (inputs_ux - inputs_lx) / 2
    error_term = torch.zeros((inputs.shape[1] * inputs.shape[2] * inputs.shape[3],
                              inputs.shape[1], inputs.shape[2], inputs.shape[3]))
    k = 0
    for i in range(INPUT_SIZE):
        for j in range(INPUT_SIZE):
            error_term[k, 0, i, j] = error_term_apt[0, 0, i, j]
            k += 1

    for layer in net.layers:
        if type(layer) is torch.nn.modules.linear.Linear:
            # print('Linear layer')
            inputs, error_term = affine_transform(layer, inputs, error_term)

        elif type(layer) is torch.nn.modules.activation.ReLU:
            # print('Relu layer')
            inputs, error_term = relu_transform(inputs, error_term)

        elif type(layer) is torch.nn.modules.flatten.Flatten:
            # print('Flatten layer')
            inputs = inputs.view(1, 1, inputs.size()[1] * inputs.size()[2] * inputs.size()[3], 1)
            error_term = error_term.view(error_term.size()[0], 1,
                                         error_term.size()[1] * error_term.size()[2] * error_term.size()[3], 1)

        elif type(layer) is torch.nn.modules.conv.Conv2d:
            # print('Conv layer')
            inputs, error_term = conv_transform(layer, inputs, error_term)

        else:
            # print('Norm layer')
            mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1))
            sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1))
            inputs = (inputs - mean) / sigma
            error_term = error_term / sigma

    # error_term_view = error_term.view((error_term.shape[0], 10)).detach().numpy().copy()
    # old version without this line:
    error_term = error_term - error_term[:, :, true_label, :].view((error_term.shape[0], 1, 1, 1))
    # error_term_view_2 = error_term.view((error_term.shape[0], 10)).detach().numpy()
    error_apt = torch.sum(torch.abs(error_term), dim=0, keepdim=True).view(inputs.size())
    inputs_ux = inputs + error_apt  # upper bound
    inputs_lx = inputs - error_apt  # lower bound
    true_label_lx = inputs_lx[0, 0, true_label, 0].detach().numpy()
    labels_ux = inputs_ux.detach().numpy()
    labels_ux = np.delete(labels_ux, [true_label])
    end_pred = time.time()
    # print("prediction time: {}".format(end_pred - start_pred))
    if (true_label_lx > labels_ux).all():
        return True
    else:
        return False


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5'],
                        required=True,
                        help='Neural network to verify.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [400, 200, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [150, 10], 10).to(DEVICE)
    elif args.net == 'conv4':
        net = Conv(DEVICE, INPUT_SIZE, [(32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    elif args.net == 'conv5':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 1, 1), (32, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
