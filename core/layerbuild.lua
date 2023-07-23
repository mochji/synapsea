--[[
	https://github.com/x-xxoa/synapsea
	core/layerbuild.lua

	MIT License
]]--

local syntable = require("core.syntable")
local layerbuild = {
	dense,
	averagePooling1D,
	averagePooling2D,
	averagePooling3D,
	maxPooling1D,
	maxPooling2D,
	maxPooling3D,
	sumPooling1D,
	sumPooling2D,
	sumPooling3D,
	averageGlobalPooling1D,
	averageGlobalPooling2D,
	averageGlobalPooling3D,
	maxGlobalPooling1D,
	maxGlobalPooling2D,
	maxGlobalPooling3D,
	sumGlobalPooling1D,
	sumGlobalPooling2D,
	sumGlobalPooling3D,
	upSample1D,
	upSample2D,
	upSample3D,
	zeroPad1D,
	zeroPad2D,
	zeroPad3D,
	crop1D,
	crop2D,
	crop3D,
	randomCrop1D,
	randomCrop2D,
	randomCrop3D,
	convolutional1D,
	convoltuional2D,
	convolutional3D,
	convolutionalTranspose1D,
	convolutionalTranspose2D,
	convolutionalTranspose3D,
	convolutionalDepthwise1D,
	convolutionalDepthwise2D,
	convolutionalSeparable1D,
	convolutionalSeparable2D,
	convolutionalSeparable3D,
	convolutionalDepthwiseSeparable1D,
	convolutionalDepthwiseSeparable2D,
	locallyConnected1D,
	locallyConnected2D,
	locallyConnected3D,
	flatten,
	reshape,
	normalize1D,
	normalize2D,
	normalize3D,
	vectorAdd1D,
	vectorAdd2D,
	vectorAdd3D,
	vectorSubtract1D,
	vectorSubtract2D,
	vectorSubtract3D,
	dot1D,
	dot2D,
	dot3D,
	vectorDivide1D,
	vectorDivide2D,
	vectorDivide3D,
	dropOut,
	uniformNoise,
	normalNoise,
	softmax,
	activate
}

--[[
	args:

	weightsInitializer,
	weightsInitParameters,
	weightsTrainable,
	biasInitializer,
	biasInitParameters,
	biasTrainable,
	inputShape,
	outputSize,
	useBias,
	usePrelu
]]--

function layerbuild.dense(args)
	local layer = {
		config = {
			activation = args.activation,
		},
		inputShape = args.inputShape,
		outputShape = {args.outputSize},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}

	if args.usePrelu then
		layer.usePrelu = true
	end

	-- initializers

	if args.weightsInitializer then
		layer.initializer.weights = {
			initializer = args.weightsInitializer,
			initializerParameters = args.weightsInitParameters
		}
	end

	if args.biasInitializer and args.useBias then
		layer.initializer.bias = {
			initializer = args.biasInitializer,
			initializerParameters = args.biasInitParameters
		}
	end

	-- trainable

	if args.weightsTrainable then
		layer.trainable.weights = true
	end

	if args.biasTrainable and args.useBias then
		layer.trainable.bias = true
	end

	-- parameters

	layer.parameters.weights = syntable.new({args.inputShape[1], args.outputSize}, 0)

	if args.useBias then
		layer.parameters.bias = 0
	end

	return layer
end

function layerbuild.averagePooling1D(args)
	local layer = {
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1},
		config = args.config
	}
end

return layerbuild
