# importing libraries 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
import numpy as np
import random
from PIL import Image
from urllib.request import urlopen
import io
 

#404 Response

resp404=["http://farm3.staticflickr.com/2892/9354564325_aa02532a2a_b.jpg","http://farm9.staticflickr.com/8321/8039607850_cf368a6ca8_b.jpg","http://farm6.staticflickr.com/5572/14664313699_1faa3118be_b.jpg","http://farm6.staticflickr.com/5530/10608593756_2340096855_b.jpg","http://farm3.staticflickr.com/2948/15188122890_9238234b40_b.jpg","http://farm9.staticflickr.com/8433/7718542740_6d6195bbdc_b.jpg","http://farm1.staticflickr.com/56/168426617_99ec2be240_b.jpg","http://farm4.staticflickr.com/3781/13381594034_01955117a6_b.jpg","http://farm4.staticflickr.com/3897/14884449807_5b8bfd6a6a_b.jpg","http://farm4.staticflickr.com/3718/10818328315_7e97a150fe_b.jpg","http://farm4.staticflickr.com/3854/14624008989_7fb3b69134_b.jpg","http://farm7.staticflickr.com/6040/6295830952_5bef03cdc7_b.jpg","http://farm1.staticflickr.com/216/449771651_971029b47b_z.jpg","http://farm8.staticflickr.com/7358/9936434983_0732337c08_b.jpg","https://farm4.staticflickr.com/3895/14784289854_ebbf1dfdcd_b.jpg","http://farm4.staticflickr.com/3195/2898811140_c351307199_z.jpg","http://farm8.staticflickr.com/7432/9616552318_8a2ce92896_b.jpg","http://farm4.staticflickr.com/3701/12800463963_4e495216a3_b.jpg "]

#INITIALIZE PARAMATERS
prob_param={'noEpoch':5,'batch_size':5,'img_width':32,'img_height':32,'trainSamples':5,'testSamples':3}
  
if K.image_data_format() == 'channels_first': 
    prob_param['input_shape'] = (3, prob_param['img_width'], prob_param['img_height']) 
else: 
    prob_param['input_shape'] = (prob_param['img_width'], prob_param['img_height'], 3)  
#CREATE MODEL
model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = prob_param['input_shape'])) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 
    
model.add(Flatten()) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 
  
model.compile(loss ='binary_crossentropy', optimizer ='rmsprop',  metrics =['accuracy']) 

print("MODEL COMPILED:",model.summary())

#LOAD DATA FROM WEB
def loadData(fileName,start=0,end=10):
	f=open(fileName)
	lines=f.readlines()
	inputs=[]
	outputs=[]
	for i in range(start,end):
		line=random.randint(0,len(lines))
		current=lines[line].split(",")
		print("Data loading....",i+1," of ",end," URL:", current[0][1:-1])
		if((not current[0][1:-1] in resp404) and (not current[0][1:-1] in inputs)):
			fd = urlopen(current[0][1:-1])
			image_file = io.BytesIO(fd.read())
			image = Image.open(image_file)
			image= image.convert('RGB')
			image=image.resize((prob_param['img_width'], prob_param['img_height']))
			array=np.asarray(image)
			inputs.append(np.asarray(image))
			outputs.append(i%2)
	print("Data load complete!!!")
	return np.array(inputs),np.array(outputs)

print("Parameters are:",prob_param,"\n") 

print("-----Loading train data-----")
trainInputs,trainOutputs=loadData("faceexp-comparison-data-train-public.csv",end=prob_param['trainSamples'])
print("-----Loading test data-----")
testInputs,testOutputs=loadData("faceexp-comparison-data-train-public.csv",end=prob_param['testSamples'])
#TRAIN MODEL
model.fit(trainInputs,trainOutputs,batch_size=prob_param["batch_size"],epochs=prob_param["noEpoch"])
print("Model TRAIN COMPLETE")

print("Predict:",model.predict_classes(testInputs))



