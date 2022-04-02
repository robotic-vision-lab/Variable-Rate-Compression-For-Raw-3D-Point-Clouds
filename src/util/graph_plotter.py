import matplotlib.pyplot as plt
def bitrate_accuracy_plotter():
    bitrate_emd = [0.070164, 0.556715, 1.094457, 1.130547, ]
    cls_acc_emd = [.916362, 0.938008, 0.943598, 0.944004, ]
    bitrate_chamfer = [0.037053, 0.277366, 1.130103, 1.431564,]
    cls_acc_chamfer = [0.933007, 0.975490, 0.973856, 0.952614,]
    emd_loss = [.5, .4, .3, .2, .1]

    fig, ax1 = plt.subplots()
    #     ax2 = ax1.twinx()
    ax1.plot(bitrate_emd, cls_acc_emd, color='red', marker='x', label='EMD')
    ax1.set_xlabel('Bitrate(Bit per point)', fontsize=12)
    #ax1.set_yticks(cls_acc_emd)
    # ax2.plot(bitrate, emd_loss, alpha=0, label ='Loss')
    # ax2.set_yticks(emd_loss)
    ax1.plot(bitrate_chamfer, cls_acc_chamfer, color='green', marker='o', label='Chamfer')
    ax1.hlines(y=.987, xmin=0, xmax=1.5, colors='purple', linestyles='dashed', label='Uncompressed')
    #plt.title('title name')

    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_ylim(.80, 1.0)
    plt.legend(loc='lower right')
    plt.show()

def var_bitrate_accuracy_plotter():
    latent_code_size = [512, 681, 684, 699, 724, 768, 896,960, 992, 1008, 1020, 1024]
    chamfer_loss = [2.283, 1.14, 0.402, 0.397, 0.3878, 0.3815, 0.3554, 0.3515, 0.3437, 0.3438, 0.3186, 0.3184 ]

    fig, ax1 = plt.subplots()
    #     ax2 = ax1.twinx()
    ax1.plot(latent_code_size[2:], chamfer_loss[2:], color='red', marker='o', label='chamfer')
    ax1.set_xlabel('Latent vector size', fontsize=12)
    ax1.set_ylabel('Chamfer loss', fontsize=12)
    plt.legend(loc='lower right')
    plt.show()

var_bitrate_accuracy_plotter()