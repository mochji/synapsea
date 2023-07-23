{
    trainingConfig = {
        optimizer = {
            optimizer = 'momentum', 
            parameters = {
                alpha = 0.9, 
                momentum = 0
            }
        }
    }, 
    metaData = {
        description = 'A simple testing NN model :3', 
        author = 'Katie', 
        name = 'Example NN'
    }, 
    inputShape = {
        1 = 10
    }, 
    layerConfig = {
        1 = {
            config = {
                stride = {
                    1 = 1
                }, 
                dilation = {
                    1 = 1
                }, 
                kernel = {
                    1 = 2
                }
            }, 
            type = 'averagePooling1D', 
            inputShape = {
                1 = 10
            }, 
            outputShape = {
                1 = 9
            }
        }, 
        2 = {
            config = {
                activation = 'leakyrelu'
            }, 
            outputShape = {
                1 = 3
            }, 
            usePrelu = true, 
            parameters = {
                alpha = 0.1, 
                weights = {
                    1 = {
                        1 = 0.12372136488969, 
                        2 = 0.025534388266951, 
                        3 = 0.0048572966969133
                    }, 
                    2 = {
                        1 = -0.11492456126302, 
                        2 = 0.060522102200547, 
                        3 = -0.0082722052274755
                    }, 
                    3 = {
                        1 = -0.046174042503575, 
                        2 = 0.0022199396075509, 
                        3 = -0.17156618452924
                    }, 
                    4 = {
                        1 = -0.046391336397788, 
                        2 = -0.10187549854371, 
                        3 = 0.081991375787738
                    }, 
                    5 = {
                        1 = 0.019725838210247, 
                        2 = 0.31443037277772, 
                        3 = -0.055782349436479
                    }, 
                    6 = {
                        1 = 0.078890752631016, 
                        2 = 0.12073603538882, 
                        3 = -0.057881639035228
                    }, 
                    7 = {
                        1 = -0.0040564152699453, 
                        2 = -0.27894213642391, 
                        3 = -0.20590725995366
                    }, 
                    8 = {
                        1 = 0.019218817451651, 
                        2 = -0.047442894558174, 
                        3 = -0.044942370304176
                    }, 
                    9 = {
                        1 = 0.014509757825199, 
                        2 = 0.13257756197239, 
                        3 = 0.18093730960896
                    }
                }, 
                bias = 0.1
            }, 
            initializer = {
                bias = {
                    initializerParameters = {
                        value = 0.1
                    }, 
                    initializer = 'constant'
                }, 
                weights = {
                    initializerParameters = {
                        mean = 0, 
                        sd = 0.1
                    }, 
                    initializer = 'normalRandom'
                }
            }, 
            type = 'dense', 
            inputShape = {
                1 = 9
            }, 
            trainable = {
                weights = true
            }
        }
    }
}
