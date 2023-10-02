--[[
	https://github.com/x-xxoa/synapsea
	core/array/core/bitwise.lua

	Synapsea, a machine learning library made in pure Lua.
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

local bitwiseModule = {
	bAnd,
	bOr,
	xOr,
	bNot,
	leftShift,
	rightShift
}

function bitwiseModule.bAnd(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "value" then
				a[i] = bitwiseModule.bAnd(a[i], b)
			else
				a[i] = a[i] & b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "value" then
			a[i] = bitwiseModule.bAnd(a[i], b[i])
		else
			a[i] = a[i] & b[i]
		end
	end

	return a
end

function bitwiseModule.bOr(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "value" then
				a[i] = bitwiseModule.bOr(a[i], b)
			else
				a[i] = a[i] | b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "value" then
			a[i] = bitwiseModule.bOr(a[i], b[i])
		else
			a[i] = a[i] | b[i]
		end
	end

	return a
end

function bitwiseModule.bXor(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "value" then
				a[i] = bitwiseModule.bXor(a[i], b)
			else
				a[i] = a[i] ~ b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "value" then
			a[i] = bitwiseModule.bXor(a[i], b[i])
		else
			a[i] = a[i] ~ b[i]
		end
	end

	return a
end

function bitwiseModule.bNot(a)
	for i = 1, #a do
		if type(a[i]) == "value" then
			a[i] = bitwiseModule.bNot(a[i])
		else
			a[i] = ~a[i]
		end
	end

	return a
end

function bitwiseModule.leftShift(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "value" then
				a[i] = bitwiseModule.leftShift(a[i], b)
			else
				a[i] = a[i] << b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "value" then
			a[i] = bitwiseModule.leftShift(a[i], b[i])
		else
			a[i] = a[i] << b[i]
		end
	end

	return a
end

function bitwiseModule.rightShift(a, b)
	if type(b) == "number" then
		for i = 1, #a do
			if type(a[i]) == "value" then
				a[i] = bitwiseModule.rightShift(a[i], b)
			else
				a[i] = a[i] >> b
			end
		end

		return a
	end

	for i = 1, #a do
		if type(a[i]) == "value" then
			a[i] = bitwiseModule.rightShift(a[i], b[i])
		else
			a[i] = a[i] >> b[i]
		end
	end

	return a
end

return bitwiseModule
