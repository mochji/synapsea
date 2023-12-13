--[[
	https://github.com/mochji/synapsea
	core/layers/build.lua

	Synapsea, a simple yet powerful machine learning framework for Lua.
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

local checkargs = require("core.utils.checkargs")
local canindex  = require("core.utils.canindex")

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
	dropout
}

function buildModule.dense(layerConfig)
	local defaults = {
		activation       = "linear",
		useBias          = false,

		weightsInit      = "constant",
		weightsInitArgs  = {value = 0},
		biasInit         = "constant",
		biasInitArgs     = {value = 0},

		weightsTrainable = false,
		biasTrainable    = false
	}

	checkargs(
		{layerConfig.inputShape, layerConfig.outputSize},
		{"inputShape",           "outputSize"},
		"dense"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			outputSize = layerConfig.outputSize
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = {layerConfig.outputSize}
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		parameters = layerConfig.weightsInitParameters
	}

	if layerConfig.useBias then
		layer.initializer.bias = {
			initializer = layerConfig.biasInit,
			parameters = layerConfig.biasInitParameters
		}
	end

	-- Trainable

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	layer.trainable.bias =
		layerConfig.biasTrainable and layerConfig.useBias or false

	-- Parameters

	parameterBuild.weights = {
		layerConfig.inputShape[1],
		layerConfig.outputSize
	}

	if layerConfig.useBias then
		layer.parameters.bias = 0
	end

	return layer, parameterBuild
end

function buildModule.averagePooling1D(layerConfig)
	local defaults = {
		kernel   = {1},
		stride   = {1},
		dilation = {1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"averagePooling1D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	return layer
end

function buildModule.averagePooling2D(layerConfig)
	local defaults = {
		kernel   = {1, 1},
		stride   = {1, 1},
		dilation = {1, 1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"averagePooling2D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	end

	return layer
end

function buildModule.averagePooling3D(layerConfig)
	local defaults = {
		kernel   = {1, 1, 1},
		stride   = {1, 1, 1},
		dilation = {1, 1, 1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"averagePooling3D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[4] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	end

	return layer
end

function buildModule.maxPooling1D(layerConfig)
	local defaults = {
		kernel   = {1},
		stride   = {1},
		dilation = {1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"maxPooling1D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	return layer
end

function buildModule.maxPooling2D(layerConfig)
	local defaults = {
		kernel   = {1, 1},
		stride   = {1, 1},
		dilation = {1, 1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"maxPooling2D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	end

	return layer
end

function buildModule.maxPooling3D(layerConfig)
	local defaults = {
		kernel   = {1, 1, 1},
		stride   = {1, 1, 1},
		dilation = {1, 1, 1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"maxPooling3D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[4] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	end

	return layer
end

function buildModule.sumPooling1D(layerConfig)
	local defaults = {
		kernel   = {1},
		stride   = {1},
		dilation = {1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"sumPooling1D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	return layer
end

function buildModule.sumPooling2D(layerConfig)
	local defaults = {
		kernel   = {1, 1},
		stride   = {1, 1},
		dilation = {1, 1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"sumPooling2D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	end

	return layer
end

function buildModule.sumPooling3D(layerConfig)
	local defaults = {
		kernel   = {1, 1, 1},
		stride   = {1, 1, 1},
		dilation = {1, 1, 1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"sumPooling3D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[4] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	end

	return layer
end

function buildModule.upSample1D(layerConfig)
	local defaults = {
		kernel = {2}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"upSample1D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	return layer
end

function buildModule.upSample2D(layerConfig)
	local defaults = {
		kernel = {2, 2}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"upSample2D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] * layerConfig.kernel[1], layerConfig.inputShape[3] * layerConfig.kernel[2]
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] * layerConfig.kernel[1], layerConfig.inputShape[2] * layerConfig.kernel[2]
		}
	end

	return layer
end

function buildModule.upSample3D(layerConfig)
	local defaults = {
		kernel = {2, 2}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"upSample3D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			kernel = layerConfig.kernel
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] * layerConfig.kernel[1],
			layerConfig.inputShape[3] * layerConfig.kernel[2],
			layerConfig.inputShape[4] * layerConfig.kernel[3]
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] * layerConfig.kernel[1],
			layerConfig.inputShape[2] * layerConfig.kernel[2],
			layerConfig.inputShape[3] * layerConfig.kernel[3]
		}
	end

	return layer
end

function buildModule.zeroPad1D(layerConfig)
	local defaults = {
		paddingAmount = {1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"zeroPad1D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			paddingAmount = layerConfig.paddingAmount
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] + layerConfig.paddingAmount[1] * 2
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] + layerConfig.paddingAmount[1] * 2
		}
	end

	return layer
end

function buildModule.zeroPad2D(layerConfig)
	local defaults = {
		paddingAmount = {1, 1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"zeroPad2D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			paddingAmount = layerConfig.paddingAmount
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] + layerConfig.paddingAmount[1] * 2,
			layerConfig.inputShape[3] + layerConfig.paddingAmount[2] * 2
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] + layerConfig.paddingAmount[1] * 2,
			layerConfig.inputShape[2] + layerConfig.paddingAmount[2] * 2
		}
	end

	return layer
end

function buildModule.zeroPad3D(layerConfig)
	local defaults = {
		paddingAmount = {1, 1, 1}
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"zeroPad3D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer = {
		config = {
			paddingAmount = layerConfig.paddingAmount
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.inputShape[2] + layerConfig.paddingAmount[1] * 2,
			layerConfig.inputShape[3] + layerConfig.paddingAmount[2] * 2,
			layerConfig.inputShape[4] + layerConfig.paddingAmount[3] * 2
		}
	else
		layer.outputShape = {
			layerConfig.inputShape[1] + layerConfig.paddingAmount[1] * 2,
			layerConfig.inputShape[2] + layerConfig.paddingAmount[2] * 2,
			layerConfig.inputShape[3] + layerConfig.paddingAmount[3] * 2
		}
	end

	return layer
end

function buildModule.crop1D(layerConfig)
	checkargs(
		{layerConfig.inputShape, layerConfig.outputShape, layerConfig.start},
		{"inputShape",           "outputShape",           "start"},
		"crop1D"
	)

	local layer = {
		config = {
			start = layerConfig.start,
			outputShape = layerConfig.outputShape
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.outputShape[1]
		}
	else
		layer.outputShape = layerConfig.outputShape
	end

	return layer
end

function buildModule.crop2D(layerConfig)
	checkargs(
		{layerConfig.inputShape, layerConfig.outputShape, layerConfig.start},
		{"inputShape",           "outputShape",           "start"},
		"crop2D"
	)

	local layer = {
		config = {
			start = layerConfig.start,
			outputShape = layerConfig.outputShape
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.outputShape[1],
			layerConfig.outputShape[2]
		}
	else
		layer.outputShape = layerConfig.outputShape
	end

	return layer
end

function buildModule.crop3D(layerConfig)
	checkargs(
		{layerConfig.inputShape, layerConfig.outputShape, layerConfig.start},
		{"inputShape",           "outputShape",           "start"},
		"crop3D"
	)

	local layer = {
		config = {
			start = layerConfig.start,
			outputShape = layerConfig.outputShape
		},
		inputShape = layerConfig.inputShape
	}

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			layerConfig.outputShape[1],
			layerConfig.outputShape[2],
			layerConfig.outputShape[3]
		}
	else
		layer.outputShape = layerConfig.outputShape
	end

	return layer
end

function buildModule.convolutional1D(layerConfig)
	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2},
		stride          = {1},
		dilation        = {1},
		filters         = 1,

		filterInit      = "constant",
		filterInitArgs  = {value = 0.1},
		biasInit        = "constant",
		biasInitArgs    = {value = 0.1},

		filterTrainable = false,
		biasTrainable   = false
	}

	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"convolutional1D"
	)

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation,
			filters = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		parameters = layerConfig.filterInitParameters
	}

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			parameters = layerConfig.biasesInitParameters
		}
	end

	-- Trainable

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	layer.trainable.bias =
		layerConfig.filterTrainable and layerConfig.useBias or false

	-- Parameters

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1]
	}

	if layerConfig.useBias then
		parameterBuild.biases = {layer.outputShape}
	end

	return layer, parameterBuild
end

function buildModule.convolutional2D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"convolutional2D"
	)

	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2, 2},
		stride          = {1, 1},
		dilation        = {1, 1},
		filters         = 1,

		filterInit      = "constant",
		filterInitArgs  = {value = 0.1},
		biasInit        = "constant",
		biasInitArgs    = {value = 0.1},

		filterTrainable = false,
		biasTrainable   = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation,
			filters = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[3] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		parameters = layerConfig.filterInitParameters
	}

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			parameters = layerConfig.biasesInitParameters
		}
	end

	-- Trainable

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	layer.trainable.biases =
		layerConfig.biasesTrainable and layerConfig.useBias or false

	-- Parameters

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1],
		layerConfig.kernel[2]
	}

	if layerConfig.useBias then
		parameterBuild.biases = {layer.outputShape}
	end

	return layer, parameterBuild
end

function buildModule.convolutional3D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"convolutional3D"
	)

	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2, 2, 2},
		stride          = {1, 1, 1},
		dilation        = {1, 1, 1},
		filters         = 1,

		filterInit      = "constant",
		filterInitArgs  = {value = 0.1},
		biasInit        = "constant",
		biasInitArgs    = {value = 0.1},

		filterTrainable = false,
		biasTrainable   = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation,
			filters = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[4] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[4] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	else
		layer.outputShape = {
			math.floor((layerConfig.inputShape[1] - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor((layerConfig.inputShape[2] - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor((layerConfig.inputShape[3] - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		parameters = layerConfig.filterInitParameters
	}

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			parameters = layerConfig.biasesInitParameters
		}
	end

	-- Trainable

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	layer.trainable.biases =
		layerConfig.biasesTrainable and layerConfig.useBias or false

	-- Parameters

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1],
		layerConfig.kernel[2],
		layerConfig.kernel[3]
	}

	if layerConfig.useBias then
		parameterBuild.biases = {layer.outputShape}
	end

	return layer, parameterBuild
end

function buildModule.convolutionalTranspose1D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"convolutionalTranspose1D"
	)

	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2},
		stride          = {1},
		dilation        = {1},
		paddingAmount   = {1},
		filters         = 1,

		filterInit      = "constant",
		filterInitArgs  = {value = 0.1},
		biasInit        = "constant",
		biasInitArgs    = {value = 0.1},

		filterTrainable = false,
		biasTrainable   = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation,
			filters = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor(((layerConfig.inputShape[2] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	else
		layer.outputShape = {
			math.floor(((layerConfig.inputShape[1] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1
		}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		parameters = layerConfig.filterInitParameters
	}

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			parameters = layerConfig.biasesInitParameters
		}
	end

	-- Trainable

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	layer.trainable.bias =
		layerConfig.filterTrainable and layerConfig.useBias or false

	-- Parameters

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1]
	}

	if layerConfig.useBias then
		parameterBuild.biases = {layer.outputShape}
	end

	return layer, parameterBuild
end

function buildModule.convolutionalTranspose2D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"convolutionalTranspose2D"
	)

	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2, 2},
		stride          = {1, 1},
		dilation        = {1, 1},
		paddingAmount   = {1, 1},
		filters         = 1,

		filterInit      = "constant",
		filterInitArgs  = {value = 0.1},
		biasInit        = "constant",
		biasInitArgs    = {value = 0.1},

		filterTrainable = false,
		biasTrainable   = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation,
			filters = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor(((layerConfig.inputShape[2] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor(((layerConfig.inputShape[3] + layerConfig.paddingAmount[2]) - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	else
		layer.outputShape = {
			math.floor(((layerConfig.inputShape[1] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor(((layerConfig.inputShape[2] + layerConfig.paddingAmount[2]) - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1
		}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		parameters = layerConfig.filterInitParameters
	}

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			parameters = layerConfig.biasesInitParameters
		}
	end

	-- Trainable

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	layer.trainable.bias =
		layerConfig.filterTrainable and layerConfig.useBias or false

	-- Parameters

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1],
		layerConfig.kernel[2]
	}

	if layerConfig.useBias then
		parameterBuild.biases = {layer.outputShape}
	end

	return layer, parameterBuild
end

function buildModule.convolutionalTranspose3D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"convolutionalTranspose3D"
	)

	local defaults = {
		activation      = "linear",
		useBias         = false,

		kernel          = {2, 2, 2},
		stride          = {1, 1, 1},
		dilation        = {1, 1, 1},
		paddingAmount   = {1, 1, 1},
		filters         = 1,

		filterInit      = "constant",
		filterInitArgs  = {value = 0.1},
		biasInit        = "constant",
		biasInitArgs    = {value = 0.1},

		filterTrainable = false,
		biasTrainable   = false
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		config = {
			activation = layerConfig.activation,
			stride = layerConfig.stride,
			dilation = layerConfig.dilation,
			filters = layerConfig.filters
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape
	}, {}

	-- Output shape

	if layerConfig.inputShape[2] then
		layer.outputShape = {
			layerConfig.inputShape[1],
			math.floor(((layerConfig.inputShape[2] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor(((layerConfig.inputShape[3] + layerConfig.paddingAmount[2]) - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor(((layerConfig.inputShape[4] + layerConfig.paddingAmount[3]) - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	else
		layer.outputShape = {
			math.floor(((layerConfig.inputShape[1] + layerConfig.paddingAmount[1]) - layerConfig.kernel[1]) / layerConfig.stride[1]) + 1,
			math.floor(((layerConfig.inputShape[2] + layerConfig.paddingAmount[2]) - layerConfig.kernel[2]) / layerConfig.stride[2]) + 1,
			math.floor(((layerConfig.inputShape[3] + layerConfig.paddingAmount[3]) - layerConfig.kernel[3]) / layerConfig.stride[3]) + 1
		}
	end

	-- Initializers

	layer.initializer.filter = {
		initializer = layerConfig.filterInit,
		parameters = layerConfig.filterInitParameters
	}

	if layerConfig.useBias then
		layer.initializer.biases = {
			initializer = layerConfig.biasesInit,
			parameters = layerConfig.biasesInitParameters
		}
	end

	-- Trainable

	layer.trainable.filter =
		layerConfig.filterTrainable and true or false

	layer.trainable.bias =
		layerConfig.filterTrainable and layerConfig.useBias or false

	-- Parameters

	parameterBuild.filter = {
		layerConfig.filters,
		layerConfig.kernel[1],
		layerConfig.kernel[2],
		layerConfig.kernel[3]
	}

	if layerConfig.useBias then
		parameterBuild.biases = {layer.outputShape}
	end

	return layer, parameterBuild
end

function buildModule.flatten(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"flatten"
	)

	local outputShape = 1

	for a = 1, #layerConfig.inputShape do
		outputShape = outputShape * layerConfig.inputShape[a]
	end

	return {
		inputShape = layerConfig.inputShape,
		outputShape = {outputShape}
	}
end

function buildModule.add1D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"add1D"
	)

	local defaults = {
		biasesInit     = "constant",
		biasesInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = layerConfig.biasesInit,
		parameters = layerConfig.biasesInitParameters
	}

	-- Trainable

	layer.trainable.bias =
		layerConfig.biasTrainable and true or false

	-- Parameters

	parameterBuild.biases = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.add2D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"add2D"
	)

	local defaults = {
		biasesInit     = "constant",
		biasesInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = layerConfig.biasesInit,
		parameters = layerConfig.biasesInitParameters
	}

	-- Trainable

	layer.trainable.bias =
		layerConfig.biasTrainable and true or false

	-- Parameters

	parameterBuild.biases = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.add3D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"add3D"
	)

	local defaults = {
		biasesInit     = "constant",
		biasesInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = layerConfig.biasesInit,
		parameters = layerConfig.biasesInitParameters
	}

	-- Trainable

	layer.trainable.bias =
		layerConfig.biasTrainable and true or false

	-- Parameters

	parameterBuild.biases = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.subtract1D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"subtract1D"
	)

	local defaults = {
		biasesInit     = "constant",
		biasesInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = layerConfig.biasesInit,
		parameters = layerConfig.biasesInitParameters
	}

	-- Trainable

	layer.trainable.bias =
		layerConfig.biasTrainable and true or false

	-- Parameters

	parameterBuild.biases = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.subtract2D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"subtract2D"
	)

	local defaults = {
		biasesInit     = "constant",
		biasesInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = layerConfig.biasesInit,
		parameters = layerConfig.biasesInitParameters
	}

	-- Trainable

	layer.trainable.bias =
		layerConfig.biasTrainable and true or false

	-- Parameters

	parameterBuild.biases = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.subtract3D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"subtract3D"
	)

	local defaults = {
		biasesInit     = "constant",
		biasesInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.biases = {
		initializer = layerConfig.biasesInit,
		parameters = layerConfig.biasesInitParameters
	}

	-- Trainable

	layer.trainable.bias =
		layerConfig.biasTrainable and true or false

	-- Parameters

	parameterBuild.biases = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.multiply1D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"multiply1D"
	)

	local defaults = {
		weightsInit     = "constant",
		weightsInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		parameters = layerConfig.weightsInitParameters
	}

	-- Trainable

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	-- Parameters

	parameterBuild.weights = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.multiply2D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"multiply2D"
	)

	local defaults = {
		weightsInit     = "constant",
		weightsInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		parameters = layerConfig.weightsInitParameters
	}

	-- Trainable

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	-- Parameters

	parameterBuild.weights = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.multiply3D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"multiply3D"
	)

	local defaults = {
		weightsInit     = "constant",
		weightsInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		parameters = layerConfig.weightsInitParameters
	}

	-- Trainable

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	-- Parameters

	parameterBuild.weights = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.divide1D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"divide1D"
	)

	local defaults = {
		weightsInit     = "constant",
		weightsInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		parameters = layerConfig.weightsInitParameters
	}

	-- Trainable

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	-- Parameters

	parameterBuild.weights = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.divide2D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"divide2D"
	)

	local defaults = {
		weightsInit     = "constant",
		weightsInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		parameters = layerConfig.weightsInitParameters
	}

	-- Trainable

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	-- Parameters

	parameterBuild.weights = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.divide3D(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"divide3D"
	)

	local defaults = {
		weightsInit     = "constant",
		weightsInitArgs = {value = 0}
	}

	layerConfig = setmetatable(layerConfig, {__index = defaults})

	local layer, parameterBuild = {
		parameters = {},
		trainable = {},
		initializer = {},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}, {}

	-- Initializers

	layer.initializer.weights = {
		initializer = layerConfig.weightsInit,
		parameters = layerConfig.weightsInitParameters
	}

	-- Trainable

	layer.trainable.weights =
		layerConfig.weightsTrainable and true or false

	-- Parameters

	parameterBuild.weights = {layerConfig.inputShape}

	return layer, parameterBuild
end

function buildModule.softmax(layerConfig)
	checkargs(
		{layerConfig.inputShape},
		{"inputShape"},
		"softmax"
	)

	return {
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}
end

function buildModule.activate(layerConfig)
	checkargs(
		{layerConfig.inputShape, layerConfig.activation},
		{"inputShape",           "activation"},
		"activate"
	)

	return {
		config = {
			activation = layerConfig.activation,
			derivative =
				layerConfig.derivative and true or false
		},
		parameters = {
			alpha = layerConfig.alpha
		},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}
end


function buildModule.dropout(layerConfig)
	checkargs(
		{layerConfig.inputShape, layerConfig.rate},
		{"inputShape",           "rate"},
		"dropout"
	)

	return {
		config = {
			rate = layerConfig.rate
		},
		inputShape = layerConfig.inputShape,
		outputShape = layerConfig.inputShape
	}
end

return buildModule
