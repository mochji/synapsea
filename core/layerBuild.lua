--[[
	https://github.com/x-xxoa/synapsea
	core/layerBuild.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
	Copyright (C) 2023 x-xxoa
																		   
	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.
																		   
	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
																		   
	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <https://www.gnu.org/licenses/>.
]]--

local layerBuildModule = {
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
	convolutional1D,
	convolutional2D,
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
	minMaxNormalize1D,
	minMaxNormalize2D,
	minMaxNormalize3D,
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
	add1D,
	add2D,
	add3D,
	subtract1D,
	subtract2D,
	subtract3D,
	multiply1D,
	multiply2D,
	multiply3D,
	divide1D,
	divide2D,
	divide3D,
	dropOut,
	uniformNoise,
	normalNoise,
	softmax,
	activate
}

function layerBuildModule.dense(args)
	-- Default values

	args.activation = args.activation or "linear"

	local layer, parameterBuild = {
		config = {
			activation = args.activation,
			outputSize = args.outputSize
		},
		inputShape = args.inputShape,
		outputShape = {args.outputSize},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}, {}

	if args.usePrelu and args.alpha then
		layer.usePrelu = true
	end

	-- Initializers

	if args.weightsInitializer then
		layer.initializer.weights = {
			initializer = args.weightsInitializer,
			parameters = args.weightsInitParameters
		}
	end

	if args.biasInitializer and args.useBias then
		layer.initializer.bias = {
			initializer = args.biasInitializer,
			parameters = args.biasInitParameters
		}
	end

	-- Trainable

	if args.weightsTrainable then
		layer.trainable.weights = true
	end

	if args.biasTrainable and args.useBias then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.weights = {
		shape = {args.inputShape[1], args.outputSize}
	}

	if args.useBias then
		layer.parameters.bias = 0
	end

	return layer, parameterBuild
end

function layerBuildModule.averagePooling1D(args)
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

function layerBuildModule.averagePooling2D(args)
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

function layerBuildModule.averagePooling3D(args)
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

layerBuildModule.maxPooling1D = layerBuildModule.averagePooling1D -- They do the same stuff so we can just do this

layerBuildModule.maxPooling2D = layerBuildModule.averagePooling2D

layerBuildModule.maxPooling3D = layerBuildModule.averagePooling3D

layerBuildModule.sumPooling1D = layerBuildModule.averagePooling1D

layerBuildModule.sumPooling2D = layerBuildModule.averagePooling2D

layerBuildModule.sumPooling3D = layerBuildModule.averagePooling3D

function layerBuildModule.averageGlobalPooling1D(args)
	return {
		inputShape = args.inputShape,
		outputShape = {1}
	}
end

function layerBuildModule.averageGlobalPooling2D(args)
	return {
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1]}
	}
end

layerBuildModule.averageGlobalPooling3D = layerBuildModule.averageGlobalPooling2D

layerBuildModule.maxGlobalPooling1D = layerBuildModule.averageGlobalPooling1D

layerBuildModule.maxGlobalPooling2D = layerBuildModule.averageGlobalPooling2D

layerBuildModule.maxGlobalPooling3D = layerBuildModule.averageGlobalPooling3D

layerBuildModule.sumGlobalPooling1D = layerBuildModule.averageGlobalPooling1D

layerBuildModule.sumGlobalPooling2D = layerBuildModule.averageGlobalPooling2D

layerBuildModule.sumGlobalPooling3D = layerBuildModule.averageGlobalPooling3D

function layerBuildModule.upSample1D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1]}
	}
end

function layerBuildModule.upSample2D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2]}
	}
end

function layerBuildModule.upSample3D(args)
	return {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2], args.inputShape[3] * args.kernel[3]}
	}
end

function layerBuildModule.zeroPad1D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2}
	}
end

function layerBuildModule.zeroPad2D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2}
	}
end

function layerBuildModule.zeroPad3D(args)
	return {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2, args.inputShape[3] + args.paddingAmount[3] * 2}
	}
end

function layerBuildModule.crop1D(args)
	return {
		config = {
			start = args.start,
			outputShape = args.outputShape
		},
		inputShape = args.inputShape,
		outputShape = args.outputShape
	}
end

layerBuildModule.crop2D = layerBuildModule.crop1D

layerBuildModule.crop3D = layerBuildModule.crop1D

function layerBuildModule.convolutional1D(args)
	-- Default values

	args.stride = args.stride or {1}
	args.dilation = args.dilation or {1}
	args.filters = args.filters or 1
	args.activation = args.activation or "linear"

	local layer, parameterBuild = {
		config = {
			activation = args.activation,
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}, {}

	-- Initializers

	if args.filterInitializer then
		layer.initializer.filter = {
			initializer = args.filterInitializer,
			parameters = args.filterInitParameters
		}
	end

	if args.biasesInitializer and args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer,
			parameters = args.biasesInitParameters
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function layerBuildModule.convolutional2D(args)
	-- Default values

	args.stride = args.stride or {1, 1}
	args.dilation = args.dilation or {1, 1}
	args.filters = args.filters or 1
	args.activation = args.activation or "linear"

	local layer, parameterBuild = {
		config = {
			activation = args.activation,
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}, {}

	-- Initializers

	if args.filterInitializer then
		layer.initializer.filter = {
			initializer = args.filterInitializer,
			parameters = args.filterInitParameters
		}
	end

	if args.biasesInitializer and args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer,
			parameters = args.biasesInitParameters
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1], args.kernel[2]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function layerBuildModule.convolutional3D(args)
	-- Default values

	args.stride = args.stride or {1, 1, 1}
	args.dilation = args.dilation or {1, 1, 1}
	args.filters = args.filters or 1
	args.activation = args.activation or "linear"

	local layer, parameterBuild = {
		config = {
			activation = args.activation,
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}, {}

	-- Initializers

	if args.filterInitializer then
		layer.initializer.filter = {
			initializer = args.filterInitializer,
			parameters = args.filterInitParameters
		}
	end

	if args.biasesInitializer and args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer,
			parameters = args.biasesInitParameters
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1], args.kernel[2], args.kernel[3]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function layerBuildModule.convolutionalTranspose1D(args)
	-- Default values

	args.stride = args.stride or {1}
	args.dilation = args.dilation or {1}
	args.filters = args.filters or 1
	args.activation = args.activation or "linear"

	local layer, parameterBuild = {
		config = {
			activation = args.activation,
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters,
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] + args.paddingAmount[1] - args.kernel[1]) / args.stride[1]) + 1},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}, {}

	-- Initializers

	if args.filterInitializer then
		layer.initializer.filter = {
			initializer = args.filterInitializer,
			parameters = args.filterInitParameters
		}
	end

	if args.biasesInitializer and args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer,
			parameters = args.biasesInitParameters
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function layerBuildModule.convolutionalTranspose2D(args)
	-- Default values

	args.stride = args.stride or {1, 1}
	args.dilation = args.dilation or {1, 1}
	args.filters = args.filters or 1
	args.activation = args.activation or "linear"

	local layer, parameterBuild = {
		config = {
			activation = args.activation,
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters,
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] + args.paddingAmount[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] + args.paddingAmount[2] - args.kernel[2]) / args.stride[2]) + 1},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}, {}

	-- Initializers

	if args.filterInitializer then
		layer.initializer.filter = {
			initializer = args.filterInitializer,
			parameters = args.filterInitParameters
		}
	end

	if args.biasesInitializer and args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer,
			parameters = args.biasesInitParameters
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters, args.kernel[1], args.kernel[2]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

function layerBuildModule.convolutionalTranspose3D(args)
	-- Default values

	args.stride = args.stride or {1, 1, 1}
	args.dilation = args.dilation or {1, 1, 1}
	args.filters = args.filters or 1
	args.activation = args.activation or "linear"

	local layer, parameterBuild = {
		config = {
			activation = args.activation,
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters,
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape,
		outputShape = {math.floor((args.inputShape[1] + args.paddingAmount[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1},
		trainable = {},
		initializer = {},
		parameters = {
			alpha = args.alpha
		}
	}, {}

	-- Initializers

	if args.filterInitializer then
		layer.initializer.filter = {
			initializer = args.filterInitializer,
			parameters = args.filterInitParameters
		}
	end

	if args.biasesInitializer and args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInitializer,
			parameters = args.biasesInitParameters
		}
	end

	-- Trainable

	if args.filterTrainable then
		layer.trainable.filter = true
	end

	if args.biasesTrainable and args.useBias then
		layer.trainable.biases = true
	end

	-- Parameters

	parameterBuild.filter = {
		shape = {args.filters or 1, args.kernel[1], args.kernel[2], args.kernel[3]}
	}

	if args.useBias then
		parameterBuild.biases = {
			shape = layer.outputShape
		}
	end

	return layer, parameterBuild
end

--[[
layerbuild.convolutionalDepthwise1D = layerbuild.convolutional1D

layerbuild.convolutionalDepthwise2D = layerbuild.convolutional2D

layerbuild.convolutionalSeparable1D = layerbuild.convolutional1D

layerbuild.convolutionalSeparable2D = layerbuild.convolutional2D

layerbuild.convolutionalSeparable3D = layerbuild.convolutional3D

layerbuild.convolutionalDepthwiseSeparable1D = layerbuild.convolutional2D

layerbuild.convolutionalDepthwiseSeparable2D = layerbuild.convolutional3D
]]--

function layerBuildModule.flatten(args)
	local outputShape = 1

	for a = 1, #args.inputShape do
		outputShape = outputShape * args.inputShape[a]
	end

	return {
		inputShape = args.inputShape,
		outputShape = {outputShape}
	}
end

function layerBuildModule.minMaxNormalize1D(args)
	return {
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

layerBuildModule.normalize2D = layerBuildModule.normalize1D

layerBuildModule.normalize3D = layerBuildModule.normalize3D

function layerBuildModule.vectorAdd1D(args)
	local layer, parameterBuild = {
		inputShape = args.inputShape,
		outputShape = args.inputShape,
		initializer = {},
		parameters = {}
	}, {}

	-- Initializers

	if args.biasesInitializer then
		layer.initializer.biases = {
			initializer = args.biasesInitializer,
			parameters = args.biasesInitParameters
		}
	end

	-- Parameters

	parameterBuild.biases = {
		shape = args.inputShape
	}

	return layer, parameterBuild
end

layerBuildModule.vectorAdd2D = layerBuildModule.vectorAdd1D

layerBuildModule.vectorAdd3D = layerBuildModule.vectorAdd1D

layerBuildModule.vectorSubtract1D = layerBuildModule.vectorAdd1D

layerBuildModule.vectorSubtract2D = layerBuildModule.vectorAdd1D

layerBuildModule.vectorSubtract3D = layerBuildModule.vectorAdd1D

function layerBuildModule.dot1D(args)
	local layer, parameterBuild = {
		inputShape = args.inputShape,
		outputShape = args.inputShape,
		initializer = {},
		parameters = {}
	}, {}

	-- Initializers

	if args.weightsInitializer then
		layer.initializer.weights = {
			initializer = args.weightsInitializer,
			parameters = args.weightsInitParameters
		}
	end

	-- Parameters

	parameterBuild.weights = {
		shape = args.inputShape
	}

	return layer, parameterBuild
end

layerBuildModule.dot2D = layerBuildModule.dot1D

layerBuildModule.dot3D = layerBuildModule.dot1D

layerBuildModule.vectorDivide1D = layerBuildModule.dot1D

layerBuildModule.vectorDivide2D = layerBuildModule.dot1D

layerBuildModule.vectorDivide3D = layerBuildModule.dot1D

function layerBuildModule.add1D(args)
	local layer = {
		inputShape = args.inputShape,
		outputShape = args.inputShape,
		trainable = {},
		initializer = {},
		parameters = {
			bias = 0
		}
	}

	-- Trainable

	if args.biasTrainable then
		layer.trainable.bias = true
	end

	-- Initializers

	if args.biasInitializer then
		layer.initializer.bias = {
			initializer = args.biasInitializer,
			parameters = args.biasInitParameters
		}
	end

	return layer
end

layerBuildModule.add2D = layerBuildModule.add1D

layerBuildModule.add3D = layerBuildModule.add1D

layerBuildModule.subtract1D = layerBuildModule.add1D

layerBuildModule.subtract2D = layerBuildModule.add1D

layerBuildModule.subtract3D = layerBuildModule.add1D

function layerBuildModule.multiply1D(args)
	local layer = {
		inputShape = args.inputShape,
		outputShape = args.inputShape,
		trainable = {},
		initializer = {},
		parameters = {
			weight = 0
		}
	}

	-- Trainable

	if args.weightTrainable then
		layer.trainable.weight = true
	end

	-- Initializers

	if args.weightInitializer then
		layer.initializer.weight = {
			initializer = args.weightInitializer,
			parameters = args.weightInitParameters
		}
	end

	return layer
end

layerBuildModule.multiply2D = layerBuildModule.multiply1D

layerBuildModule.multiply3D = layerBuildModule.multiply1D

layerBuildModule.divide1D = layerBuildModule.multiply1D

layerBuildModule.divide2D = layerBuildModule.multiply1D

layerBuildModule.divide3D = layerBuildModule.multiply1D

function layerBuildModule.dropOut(args)
	return {
		inputShape = args.inputShape,
		outputShape = args.inputShape,
		config = {
			rate = args.rate
		}
	}
end

function layerBuildModule.uniformNoise(args)
	return {
		inputShape = args.inputShape,
		outputShape = args.inputShape,
		config = {
			lowerLimit = args.lowerLimit,
			upperLimit = args.upperLimit,
			backPropOnly = args.backPropOnly
		}
	}
end

function layerBuildModule.normalNoise(args)
	return {
		inputShape = args.inputShape,
		outputShape = args.inputShape,
		config = {
			mean = args.mean,
			sd = args.sd,
			backPropOnly = args.backPropOnly
		}
	}
end

function layerBuildModule.softmax(args)
	return {
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

function layerBuildModule.activate(args)
	-- Default values

	activation = activation or "linear"

	return {
		inputShape = args.inputShape,
		outputShape = args.outputShape,
		config = {
			activation = args.activation,
			derivative = args.derivative
		},
		parameters = {
			alpha = args.alpha
		}
	}
end

return layerBuildModule
