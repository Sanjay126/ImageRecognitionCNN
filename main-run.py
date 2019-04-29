import argparse
import model
import loadDataset

def main_run(inp_path,outDir,img_size,momentum,lossfn,batchSize,numEpochs,noOfLayers,noOfWorkers,decay,learnRate,loadModel,plot,dataAug):
    datagen,X_train,X_test,Y_train,Y_test,numClasses=loadDataset.loadDataset.getData(inp_path,img_size,dataAug,loadModel)
    model_obj=model.model(numClasses,noOfLayers,X_train[0].shape,momentum,lossfn,decay,learnRate)
    model_obj.forward(X_train,Y_train,X_test,Y_test,batchSize,numEpochs,datagen,noOfWorkers,outDir=outDir,loadModel=loadModel,Plot=plot)


def __main__():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numEpochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--imgSize', type=tuple, default=(256,256), help='img_size')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--decay', type=float, default=0.06, help='Weight decay')
    parser.add_argument('--learnRate', type=float, default=0.01, help='Learning Rate')
    parser.add_argument('--batchSize', type=int, default=80, help='Training batch size')
    parser.add_argument('--noOfLayers', type=int, default=5, help='ConvLSTM hidden state size')
    parser.add_argument('--lossfn', type=str, default='categorical_crossentropy', help='loss function',choices=['categorical_crossentropy','sparse_categorical_crossentropy','poisson','mean_absolute_error','mean_squared_logarithmic_error'])
    parser.add_argument('--outDir', type=str, default='Data', help='Output directory')
    parser.add_argument('--inpDir', type=str, default='./data', help='Directory containing  dataset')
    parser.add_argument('--loadModel', type=str, default=None, help='Directory containing model.hdf5')
    parser.add_argument('--plot', type=bool, default=True, help='Path to saved model')
    parser.add_argument('--noOfWorkers', type=int, default=12, help='No. of workers for data augmentation')
    parser.add_argument('--dataAug', type=bool, default=False, help='whether to use data augmentation')
    
    args = parser.parse_args()

    numEpochs = args.numEpochs
    imgSize=args.imgSize
    momentum = args.momentum
    decay=args.decay
    learnRate=args.learnRate
    batchSize = args.batchSize
    noOfLayers = args.noOfLayers
    lossfn = args.lossfn
    outDir = args.outDir
    inpDir = args.inpDir
    plot=args.plot
    noOfWorkers=args.noOfWorkers
    loadModel = args.loadModel
    dataAug=args.dataAug
    main_run(inpDir,outDir,imgSize,momentum,lossfn,batchSize,numEpochs,noOfLayers,noOfWorkers,decay,learnRate,loadModel,plot,dataAug)

__main__()
