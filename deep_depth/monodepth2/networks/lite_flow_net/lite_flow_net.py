import math
import torch

from . import correlation  # the custom cost volume layer


Backward_tensorGrid = {}


def Backward(tensorInput, tensorFlow):
    if str(tensorFlow.size()) not in Backward_tensorGrid:
        tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
        tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

        Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
    
    tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')



class LiteFlowNet(torch.nn.Module):
    def __init__(self):
        super(LiteFlowNet, self).__init__()

        class Features(torch.nn.Module):
            def __init__(self):
                super(Features, self).__init__()

                self.moduleOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.moduleSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )
            

            def forward(self, tensorInput):
                tensorOne = self.moduleOne(tensorInput)
                tensorTwo = self.moduleTwo(tensorOne)
                tensorThr = self.moduleThr(tensorTwo)
                tensorFou = self.moduleFou(tensorThr)
                tensorFiv = self.moduleFiv(tensorFou)
                tensorSix = self.moduleSix(tensorFiv)

                return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]

        class Matching(torch.nn.Module):
            def __init__(self, intLevel):
                super(Matching, self).__init__()

                self.dblBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                

                if intLevel == 6:
                    self.moduleUpflow = None

                elif intLevel != 6:
                    self.moduleUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)

                

                if intLevel >= 4:
                    self.moduleUpcorr = None

                elif intLevel < 4:
                    self.moduleUpcorr = torch.nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)

                

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
                )
            

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

                if tensorFlow is not None:
                    tensorFlow = self.moduleUpflow(tensorFlow)
                

                if tensorFlow is not None:
                    tensorFeaturesSecond = Backward(tensorInput=tensorFeaturesSecond, tensorFlow=tensorFlow * self.dblBackward)
                

                if self.moduleUpcorr is None:
                    tensorCorrelation = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFeaturesFirst, tensorSecond=tensorFeaturesSecond, intStride=1), negative_slope=0.1, inplace=False)

                elif self.moduleUpcorr is not None:
                    tensorCorrelation = self.moduleUpcorr(torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tensorFirst=tensorFeaturesFirst, tensorSecond=tensorFeaturesSecond, intStride=2), negative_slope=0.1, inplace=False))

                

                return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(tensorCorrelation)

        class Subpixel(torch.nn.Module):
            def __init__(self, intLevel):
                super(Subpixel, self).__init__()

                self.dblBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                if intLevel != 2:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel == 2:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 130, 130, 194, 258, 386 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
                )
            

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
                tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

                if tensorFlow is not None:
                    tensorFeaturesSecond = Backward(tensorInput=tensorFeaturesSecond, tensorFlow=tensorFlow * self.dblBackward)
                

                return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(torch.cat([ tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow ], 1))

        class Regularization(torch.nn.Module):
            def __init__(self, intLevel):
                super(Regularization, self).__init__()

                self.dblBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

                self.intUnfold = [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]

                if intLevel >= 5:
                    self.moduleFeat = torch.nn.Sequential()

                elif intLevel < 5:
                    self.moduleFeat = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
                        torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                    )

                

                self.moduleMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                if intLevel >= 5:
                    self.moduleDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
                    )

                elif intLevel < 5:
                    self.moduleDist = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=([ 0, 0, 7, 5, 5, 3, 3 ][intLevel], 1), stride=1, padding=([ 0, 0, 3, 2, 2, 1, 1 ][intLevel], 0)),
                        torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=(1, [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]), stride=1, padding=(0, [ 0, 0, 3, 2, 2, 1, 1 ][intLevel]))
                    )

                

                self.moduleScaleX = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
                self.moduleScaleY = torch.nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
            # eny

            def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
                tensorDifference = (tensorFirst - Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward))
                tensorDifference = tensorDifference.pow(2.0).sum(1, True) + 1e-6
                # tensorDifference.register_hook(lambda grad: print("1 max: {}".format(grad.max())))
                # tensorDifference.register_hook(lambda grad: print("1 min: {}".format(grad.min())))
                # tensorDifference = tensorDifference.sum(1, True)
                # tensorDifference.register_hook(lambda grad: print("2 max: {}".format(grad.max())))
                # tensorDifference.register_hook(lambda grad: print("2 min: {}".format(grad.min())))
                tensorDifference = tensorDifference.sqrt() 
                # tensorDifference.register_hook(lambda grad: print("3 max: {}".format(grad.max())))
                # tensorDifference.register_hook(lambda grad: print("3 min: {}".format(grad.min())))

                tensorDist = self.moduleDist(self.moduleMain(torch.cat([ tensorDifference, tensorFlow - tensorFlow.view(tensorFlow.size(0), 2, -1).mean(2, True).view(tensorFlow.size(0), 2, 1, 1), self.moduleFeat(tensorFeaturesFirst) ], 1)))
                tensorDist = tensorDist.pow(2.0).neg()
                tensorDist = (tensorDist - tensorDist.max(1, True)[0]).exp()

                tensorDivisor = tensorDist.sum(1, True).reciprocal()

                tensorScaleX = self.moduleScaleX(tensorDist * torch.nn.functional.unfold(input=tensorFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)) * tensorDivisor
                tensorScaleY = self.moduleScaleY(tensorDist * torch.nn.functional.unfold(input=tensorFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)) * tensorDivisor

                return torch.cat([ tensorScaleX, tensorScaleY ], 1)

        self.moduleFeatures = Features()
        self.moduleMatching = torch.nn.ModuleList([ Matching(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
        self.moduleSubpixel = torch.nn.ModuleList([ Subpixel(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
        self.moduleRegularization = torch.nn.ModuleList([ Regularization(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])

        # self.load_state_dict(torch.load('./network-' + arguments_strModel + '.pytorch'))
        # Initialization
        for m in self.modules():
            classname = m.__class__.__name__
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                # # m.weight.data.normal_(0, 0.02)
                m.weight.data = torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('BatchNorm') != -1:
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    

    def forward(self, inputs):
        """
        Args:
            inputs (tensor list): [tensorFirst; tensorSecond]
                - tensorFirst (Nx3xHxW): first image
                - tensorSecond (Nx3xHxW): second image
        Returns:
            flows (tensor dict): flow at different scale
                - each element: Nx2xHxW
                - {0: flow0, 1:flow1, ...}
                    all flows are resized as raw input size
                    0-level is upsampled from the coarsest level
                
        """
        tensorFirst, tensorSecond = inputs
        # _, _, raw_h, raw_w = tensorFirst.shape
        # tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - 0.411618
        # tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - 0.434631
        # tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - 0.454253

        # tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - 0.410782
        # tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - 0.433645
        # tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - 0.452793
        tensorFeaturesFirst = self.moduleFeatures(tensorFirst)
        tensorFeaturesSecond = self.moduleFeatures(tensorSecond)

        tensorFirst = [ tensorFirst ]
        tensorSecond = [ tensorSecond ]

        for intLevel in [ 1, 2, 3, 4, 5 ]:
            tensorFirst.append(torch.nn.functional.interpolate(input=tensorFirst[-1], size=(tensorFeaturesFirst[intLevel].size(2), tensorFeaturesFirst[intLevel].size(3)), mode='bilinear', align_corners=False))
            tensorSecond.append(torch.nn.functional.interpolate(input=tensorSecond[-1], size=(tensorFeaturesSecond[intLevel].size(2), tensorFeaturesSecond[intLevel].size(3)), mode='bilinear', align_corners=False))
        

        tensorFlow = None

        flows = {}
        for cnt, intLevel in enumerate([ -1, -2, -3, -4, -5 ]):
            tensorFlow = self.moduleMatching[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
            tensorFlow = self.moduleSubpixel[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
            tensorFlow = self.moduleRegularization[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
            flows[5-cnt] = tensorFlow

        # post-processing flow
        for i in flows:
            flows[i] = flows[i] * (20.0 * (0.5 ** (i-1)))
            # _, _, out_h, out_w = flows[i].shape
            # flows[i] = torch.nn.functional.interpolate(input=flows[i], size=(raw_h, raw_w), mode='bilinear', align_corners=False)
            # flows[i][:, 0, :, :] *= float(raw_w) / float(out_w)
            # flows[i][:, 1, :, :] *= float(raw_h) / float(out_h)
        return flows
    
##########################################################

# def estimate(tensorFirst, tensorSecond):
#     assert(tensorFirst.size(1) == tensorSecond.size(1))
#     assert(tensorFirst.size(2) == tensorSecond.size(2))

#     intWidth = tensorFirst.size(2)
#     intHeight = tensorFirst.size(1)

#     assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
#     assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

#     tensorPreprocessedFirst = tensorFirst.cuda().view(1, 3, intHeight, intWidth)
#     tensorPreprocessedSecond = tensorSecond.cuda().view(1, 3, intHeight, intWidth)

#     intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
#     intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

#     tensorPreprocessedFirst = torch.nn.functional.interpolate(input=tensorPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
#     tensorPreprocessedSecond = torch.nn.functional.interpolate(input=tensorPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    

#     tensorFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
#     tensorFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

#     return tensorFlow[0, :, :, :].cpu()

