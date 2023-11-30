--[[
	https://github.com/mochji/synapsea
	core/layers/build.lua

	Synapsea, a simple yet powerful machine learning library made in pure Lua.
	Copyright (C) 2023 mochji

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

local buildModule = {
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
	flatten,
	reshape,
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
	softmax,
	activate,
	dropOut
}

function buildModule.dense(args)
	-- Default values

	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			outputSize = args.outputSize
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = {args.outputSize}
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = args.weightsInit or "constant",
		parameters = args.weightsInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.bias = {
			initializer = args.biasInit or "constant",
			parameters = args.biasInitParameters or {value = 0}
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

function buildModule.averagePooling1D(args)
	-- Default values

	args.stride = args.stride or {1}
	args.dilation = args.dilation or {1}

	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride,
			dilation = args.dilation
		},
		inputShape = args.inputShape
	}

	if args.inputShape[2] then
		layer.outputShape = {args.inputShape[1], math.floor((args.inputShape[2] - args.kernel[1]) / args.stride[1]) + 1}
	else
		layer.outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1}
	end

	return layer
end

function buildModule.averagePooling2D(args)
	-- Default values

	args.stride = args.stride or {1, 1}
	args.dilation = args.dilation or {1, 1}

	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride,
			dilation = args.dilation
		},
		inputShape = args.inputShape
	}

	if args.inputShape[3] then
		layer.outputShape = {args.inputShape[1], math.floor((args.inputShape[2] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[3] - args.kernel[2]) / args.stride[2]) + 1}
	else
		layer.outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1}
	end

	return layer
end

function buildModule.averagePooling3D(args)
	-- Default values

	args.stride = args.stride or {1, 1, 1}
	args.dilation = args.dilation or {1, 1, 1}

	local layer = {
		config = {
			kernel = args.kernel,
			stride = args.stride,
			dilation = args.dilation
		},
		inputShape = args.inputShape
	}

	if args.inputShape[4] then
		layer.outputShape = {args.inputShape[1], math.floor((args.inputShape[2] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[3] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[4] - args.kernel[3]) / args.stride[3]) + 1}
	else
		layer.outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1}
	end

	return layer
end

buildModule.maxPooling1D = buildModule.averagePooling1D

buildModule.maxPooling2D = buildModule.averagePooling2D

buildModule.maxPooling3D = buildModule.averagePooling3D

buildModule.sumPooling1D = buildModule.averagePooling1D

buildModule.sumPooling2D = buildModule.averagePooling2D

buildModule.sumPooling3D = buildModule.averagePooling3D

function buildModule.upSample1D(args)
	local layer = {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape
	}

	if args.inputShape[2] then
		layer.outputShape = {args.inputShape[1], math.floor((args.inputShape[2] - args.kernel[1]) / args.stride[1]) + 1}
	else
		layer.outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1}
	end

	return layer
end

function buildModule.upSample2D(args)
	local layer = {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape
	}

	if args.inputShape[3] then
		layer.outputShape = {args.inputShape[1], args.inputShape[2] * args.kernel[1], args.inputShape[3] * args.kernel[2]}
	else
		layer.outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2]}
	end

	return layer
end

function buildModule.upSample3D(args)
	local layer = {
		config = {
			kernel = args.kernel
		},
		inputShape = args.inputShape
	}

	if args.inputShape[4] then
		layer.outputShape = {args.inputShape[1], args.inputShape[2] * args.kernel[1], args.inputShape[3] * args.kernel[2], args.inputShape[4] * args.kernel[3]}
	else
		layer.outputShape = {args.inputShape[1] * args.kernel[1], args.inputShape[2] * args.kernel[2], args.inputShape[3] * args.kernel[3]}
	end

	return layer
end

function buildModule.zeroPad1D(args)
	local layer = {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape
	}

	if args.inputShape[2] then
		layer.outputShape = {args.inputShape[1], args.inputShape[2] + args.paddingAmount[1] * 2}
	else
		layer.outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2}
	end

	return layer
end

function buildModule.zeroPad2D(args)
	local layer = {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape
	}

	if args.inputShape[3] then
		layer.outputShape = {args.inputShape[1], args.inputShape[2] + args.paddingAmount[1] * 2, args.inputShape[3] + args.paddingAmount[2] * 2}
	else
		layer.outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2}
	end

	return layer
end

function buildModule.zeroPad3D(args)
	local layer = {
		config = {
			paddingAmount = args.paddingAmount
		},
		inputShape = args.inputShape
	}

	if args.inputShape[4] then
		layer.outputShape = {args.inputShape[1], args.inputShape[2] + args.paddingAmount[1] * 2, args.inputShape[3] + args.paddingAmount[2]* 2, args.inputShape[4] + args.paddingAmount[3] * 2}
	else
		layer.outputShape = {args.inputShape[1] + args.paddingAmount[1] * 2, args.inputShape[2] + args.paddingAmount[2] * 2, args.inputShape[3] + args.paddingAmount[3] * 2}
	end

	return layer
end

function buildModule.crop1D(args)
	local layer = {
		config = {
			start = args.start,
			outputShape = args.outputShape
		},
		inputShape = args.inputShape
	}

	if args.inputShape[2] then
		layer.outputShape = {args.inputShape[1], args.outputShape[1]}
	else
		layer.outputShape = args.outputShape
	end

	return layer
end

function buildModule.crop2D(args)
	local layer = {
		config = {
			start = args.start,
			outputShape = args.outputShape
		},
		inputShape = args.inputShape
	}

	if args.inputShape[2] then
		layer.outputShape = {args.inputShape[1], args.outputShape[1], args.outputShape[2]}
	else
		layer.outputShape = args.outputShape
	end

	return layer
end

function buildModule.crop3D(args)
	local layer = {
		config = {
			start = args.start,
			outputShape = args.outputShape
		},
		inputShape = args.inputShape
	}

	if args.inputShape[2] then
		layer.outputShape = {args.inputShape[1], args.outputShape[1], args.outputShape[2], args.outputShape[3]}
	else
		layer.outputShape = args.outputShape
	end

	return layer
end

function buildModule.convolutional1D(args)
	-- Default values

	args.stride = args.stride or {1}
	args.dilation = args.dilation or {1}
	args.filters = args.filters or 1

	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape
	}, {}

	-- Output shape

	if args.inputShape[2] then
		layer.outputShape = {args.inputShape[1], math.floor((args.inputShape[2] - args.kernel[1]) / args.stride[1]) + 1}
	else
		layer.outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInit or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInit or "constant",
			parameters = args.biasesInitParameters or {value = 0}
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

function buildModule.convolutional2D(args)
	-- Default values

	args.stride = args.stride or {1, 1}
	args.dilation = args.dilation or {1, 1}
	args.filters = args.filters or 1

	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape
	}, {}

	-- Output shape

	if args.inputShape[3] then
		layer.outputShape = {args.inputShape[1], math.floor((args.inputShape[2] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[3] - args.kernel[2]) / args.stride[2]) + 1}
	else
		layer.outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInit or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInit or "constant",
			parameters = args.biasesInitParameters or {value = 0}
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

function buildModule.convolutional3D(args)
	-- Default values

	args.stride = args.stride or {1, 1, 1}
	args.dilation = args.dilation or {1, 1, 1}
	args.filters = args.filters or 1

	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape
	}, {}

	-- Output shape

	if args.inputShape[4] then
		layer.outputShape = {args.inputShape[1], math.floor((args.inputShape[2] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[3] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[4] - args.kernel[3]) / args.stride[3]) + 1}
	else
		layer.outputShape = {math.floor((args.inputShape[1] - args.kernel[1]) / args.stride[1]) + 1, math.floor((args.inputShape[2] - args.kernel[2]) / args.stride[2]) + 1, math.floor((args.inputShape[3] - args.kernel[3]) / args.stride[3]) + 1}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInit or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInit or "constant",
			parameters = args.biasesInitParameters or {value = 0}
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

function buildModule.convolutionalTranspose1D(args)
	-- Default values

	args.stride = args.stride or {1}
	args.dilation = args.dilation or {1}
	args.filters = args.filters or 1

	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape
	}, {}

	-- Output shape

	if args.inputShape[2] then
		layer.outputShape = {args.inputShape[1], math.floor(((args.inputShape[2] + args.paddingAmount[1]) - args.kernel[1]) / args.stride[1]) + 1}
	else
		layer.outputShape = {math.floor(((args.inputShape[1] + args.paddingAmount[1]) - args.kernel[1]) / args.stride[1]) + 1}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInit or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInit or "constant",
			parameters = args.biasesInitParameters or {value = 0}
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

function buildModule.convolutionalTranspose2D(args)
	-- Default values

	args.stride = args.stride or {1, 1}
	args.dilation = args.dilation or {1, 1}
	args.filters = args.filters or 1

	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape
	}, {}

	-- Output shape

	if args.inputShape[3] then
		layer.outputShape = {args.inputShape[1], math.floor(((args.inputShape[2] + args.paddingAmount[1]) - args.kernel[1]) / args.stride[1]) + 1, math.floor(((args.inputShape[3] + args.paddingAmount[2]) - args.kernel[2]) / args.stride[2]) + 1}
	else
		layer.outputShape = {math.floor(((args.inputShape[1] + args.paddingAmount[1]) - args.kernel[1]) / args.stride[1]) + 1, math.floor(((args.inputShape[2] + args.paddingAmount[2]) - args.kernel[2]) / args.stride[2]) + 1}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInit or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInit or "constant",
			parameters = args.biasesInitParameters or {value = 0}
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

function buildModule.convolutionalTranspose3D(args)
	-- Default values

	args.stride = args.stride or {1, 1, 1}
	args.dilation = args.dilation or {1, 1, 1}
	args.filters = args.filters or 1

	local layer, parameterBuild = {
		config = {
			activation = args.activation or "linear",
			stride = args.stride,
			dilation = args.dilation,
			filters = args.filters
		},
		parameters = {
			alpha = args.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape
	}, {}

	-- Output shape

	if args.inputShape[4] then
		layer.outputShape = {args.inputShape[1], math.floor(((args.inputShape[2] + args.paddingAmount[1]) - args.kernel[1]) / args.stride[1]) + 1, math.floor(((args.inputShape[3] + args.paddingAmount[2]) - args.kernel[2]) / args.stride[2]) + 1, math.floor(((args.inputShape[4] + args.paddingAmount[3]) - args.kernel[3]) / args.stride[3]) + 1}
	else
		layer.outputShape = {math.floor(((args.inputShape[1] + args.paddingAmount[1]) - args.kernel[1]) / args.stride[1]) + 1, math.floor(((args.inputShape[2] + args.paddingAmount[2]) - args.kernel[2]) / args.stride[2]) + 1, math.floor(((args.inputShape[3] + args.paddingAmount[3]) - args.kernel[3]) / args.stride[3]) + 1}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = args.filterInit or "constant",
		parameters = args.filterInitParameters or {value = 0}
	}

	if args.useBias then
		layer.initializer.biases = {
			initializer = args.biasesInit or "constant",
			parameters = args.biasesInitParameters or {value = 0}
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

function buildModule.flatten(args)
	local outputShape = 1

	for a = 1, #args.inputShape do
		outputShape = outputShape * args.inputShape[a]
	end

	return {
		inputShape = args.inputShape,
		outputShape = {outputShape}
	}
end

function buildModule.add1D(args)
	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = args.biasesInit or "constant",
		parameters = args.biasesInitParameters or {value = 0}
	}

	-- Trainable

	if args.biasTrainable then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.biases = {
		shape = args.inputShape
	}

	return layer, parameterBuild
end

buildModule.add2D = buildModule.add1D

buildModule.add3D = buildModule.add1D

buildModule.subtract1D = buildModule.add1D

buildModule.subtract2D = buildModule.add1D

buildModule.subtract3D = buildModule.add1D

function buildModule.multiply1D(args)
	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = args.weightsInit or "constant",
		parameters = args.weightsInitParameters or {value = 0}
	}

	-- Trainable

	if args.biasTrainable then
		layer.trainable.bias = true
	end

	-- Parameters

	parameterBuild.weights = {
		shape = args.inputShape
	}

	return layer, parameterBuild
end

buildModule.multiply2D = buildModule.multiply1D

buildModule.multiply3D = buildModule.multiply1D

buildModule.divide1D = buildModule.multiply1D

buildModule.divide2D = buildModule.multiply1D

buildModule.divide3D = buildModule.multiply1D

function buildModule.softmax(args)
	return {
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

function buildModule.activate(args)
	return {
		config = {
			activation = args.activation,
			derivative = args.derivative
		},
		parameters = {
			alpha = args.alpha
		},
		inputShape = args.inputShape,
		outputShape = args.outputShape
	}
end


function buildModule.dropOut(args)
	return {
		config = {
			rate = args.rate
		},
		inputShape = args.inputShape,
		outputShape = args.inputShape
	}
end

return buildModule
