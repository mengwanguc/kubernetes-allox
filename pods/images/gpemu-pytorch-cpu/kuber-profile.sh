# models=("resnet50" "vgg11" "alexnet" "mobilenet_v3_small" "mobilenet_v3_large")
models=("alexnet" "densenet121" "densenet161" "densenet169" "densenet201" "googlenet" "inception_v3"
        "mnasnet0_5" "mnasnet0_75" "mnasnet1_3" "mobilenet_v2" "mobilenet_v3_large" "mobilenet_v3_small"
        "resnet101" "resnet152" "resnet18" "resnet34" "resnet50" "resnext101_32x8d" "resnext50_32x4d"
        "shufflenet_v2_x0_5" "shufflenet_v2_x1_0" "shufflenet_v2_x1_5" "shufflenet_v2_x2_0" "squeezenet1_0"
        "squeezenet1_1" "vgg11" "vgg11_bn" "vgg13" "vgg13_bn" "vgg16" "vgg16_bn" "vgg19" "vgg19_bn" "wide_resnet101_2"
        "wide_resnet50_2")
batch_size=64

length=${#models[@]}

for ((i=0; i<length; i++))
do
    model=${models[i]}
    python imagenet-cpu.py --epoch=1 --workers=4 --arch=$model -b=$batch_size --kuber-profiling=True /data/test-accuracy/imagenette2
done

