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
			activation = args.activation
		},
		inputShape = args.inputShape,
		outputShape = {args.outputSize},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}

	if args.usePrelu and args.alpha then
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
		config = {
			kernel = args.kernel,
			stride = args.stride
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1}
	}

	if not args.dilation then
		layer.config.dilation = {1}
	else
		layer.config.dilation = args.dilation
	end

	return layer
end

function layerbuild.averagePooling2D(args)
	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1}
	}

	if not args.dilation then
		layer.config.dilation = {1, 1}
	else
		layer.config.dilation = args.dilation
	end

	return layer
end

function layerbuild.averagePooling3D(args)
	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1}
	}

	if not args.dilation then
		layer.config.dilation = {1, 1, 1}
	else
		layer.config.dilation = args.dilation
	end

	return layer
end

layerbuild.maxPooling1D = layerbuild.averagePooling1D -- they do the same stuff so we can just do this

layerbuild.maxPooling2D = layerbuild.averagePooling2D

layerbuild.maxPooling3D = layerbuild.averagePooling3D

layerbuild.sumPooling1D = layerbuild.averagePooling1D

layerbuild.sumPooling2D = layerbuild.averagePooling2D

layerbuild.sumPooling3D = layerbuild.averagePooling3D

function layerbuild.averageGlobalPooling1D(args)
	return {
		inputShape = args.inputShape,
		outputShape = {1}
	}
end

function layerbuild.averageGlobalPooling2D(args)
	return {
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1]}
	}
end

layerbuild.averageGlobalPooling3D = layerbuild.averageGlobalPooling2D

layerbuild.maxGlobalPooling1D = layerbuild.averageGlobalPooling1D

layerbuild.maxGlobalPooling2D = layerbuild.averageGlobalPooling2D

layerbuild.maxGlobalPooling3D = layerbuild.averageGlobalPooling3D

layerbuild.sumGlobalPooling1D = layerbuild.averageGlobalPooling1D

layerbuild.sumGlobalPooling2D = layerbuild.averageGlobalPooling2D

layerbuild.sumGlobalPooling3D = layerbuild.averageGlobalPooling3D

function layerbuild.upSample1D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1]}
	}
end

function layerbuild.upSample2D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2]}
	}
end

function layerbuild.upSample3D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2], args.inputShape[3] * args.kernel[3]}
	}
end

function layerbuild.zeroPad1D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2}
	}
end

function layerbuild.zeroPad2D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2}
	}
end

function layerbuild.zeroPad3D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2, args.inputShape[3] + args.paddingAmount[3] * 2}
	}
end

return layerbuild
