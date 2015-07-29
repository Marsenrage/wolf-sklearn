import sys
import arff
import copy
import scipy
import sklearn
import argparse
import warnings
import numpy as np
from sklearn import *
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import *
from sklearn import cross_validation
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

##############################################################################################################
# Function to convert strings in raw data to int and return a list and its element are 
# [processed data matrix, training data, label, distigunish value in label,
# number of #distigunish value in label, the original dataset]
def process_data(filename):
	raw_data = arff.load(open(filename, 'rb'))
	#print raw_data
        raw_train_original = raw_data[u'data']
        raw_train = copy.deepcopy(raw_train_original)
	raw_train_y = copy.deepcopy(raw_train_original)
      	train_rows = len(raw_train)
	train_cols = len(raw_train[0])
	#print train_cols
	train = np.zeros((train_rows, train_cols))
	L = []    
	for j in range (train_cols):
		if type(raw_train[0][j]) is unicode:
			temp = ["" for x in range(train_rows)]
			for i in range(train_rows):
				temp[i] = raw_train[i][j]
			unique_item = np.unique(temp).tolist()
			L.append(unique_item)		
		else:
			L.append(0)
	#print temp
	for j in range (train_cols):
		for i in range (train_rows):		
			if type(raw_train[i][j]) is unicode:
				raw_train[i][j] = float(L[j].index(str(raw_train[i][j])))
				#print(raw_train[i][j], type(raw_train[i][j]))	
			else:
				float(raw_train[i][j])			
	for j in range (train_cols):
		for i in range (train_rows):		
			train[i, j] = raw_train[i][j]

	#print L
        x = np.delete(train, train_cols-1, 1)
        #print x
	       
        y1 = train[:, train_cols-1]
        if type(raw_train_original[0][train_cols-1]) is unicode:
                y = ["" for i in range(train_rows)]
                for j in range(train_rows):
                        y[j] = raw_train_original[j][train_cols-1] 

        else:
                y = np.zeros((train_rows, 1))
                for j in range(train_rows):
                        y[j, 0] = raw_train_original[j][train_cols-1]
        #print type(y)
        #print y
	return [train, x, y, y1]
#####################################################################################################################
#command line argument configuration
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument("-T", type=str, nargs= 1, dest = 'train', required = True,
                   help="training data set")
parser.add_argument("-t", type=str, nargs= 1, dest = 'test', required = True,
                    help="testing data set")
parser.add_argument("-O", type=str, nargs= 1, dest = 'out_file',required = True,
                    help="output predictions file name")
parser.add_argument("-w", type=str, choices=['svm', 'rfc', 'nb'],
                   nargs = 1, required = True, dest ='classifiers', help="classifier to use.")
parser.add_argument("-o", nargs = '*', type = str, dest = 'para',
                    help="classifier options")
args = parser.parse_args()
######################################################################################################################
# obtain train and test data
train = process_data(args.train[0])
test = process_data(args.test[0])
x_train = train[1]
y_train = train[2]
y1_train = train[3]
#print y_train
x_test = test[1]
y_test = test[2]
y1_test = test[3]
#print y_test

######################################################################################################################
#classifiers
if args.classifiers[0] == 'svm':
    if args.para == None:
        print 'Warning: Option -o is required'
        exit()
    if args.para == []:
        print 'Warning: Option -o must have two arguments: kernal and Penalty parameter C of the error term'
        exit()
    try:
        float(args.para[1])
    except ValueError:
        print 'Warning: The Penalty parameter C of the error term (seconed argument of -o) must be a number!'
    if args.para[0] not in ['linear', 'rbf']:
        print 'kernal must be linear or rbf'
    else:
        clf = svm.SVC(C=float(args.para[1]), cache_size=40, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
	      kernel=args.para[0], max_iter=-1, probability=True, random_state=1,
	      shrinking=True, tol=0.001, verbose=False)
        #print clf
if args.classifiers[0] == 'rfc':
    if args.para == None:
        print 'Warning: Option -o is required'
        exit()
    if args.para == []:
        print 'Warning: Option -o must have two arguments: number of estimators and max_depth'
        exit()
    try:
        int(args.para[0])
    except ValueError:
        print 'Warning: Estimators (first argument of -o) must be a number!'
    try:
        int(args.para[1])
    except ValueError:
        print 'Warning: max_depth (seconed argument of -o) must be a number!'
    
    clf = RandomForestClassifier(n_estimators=int(args.para[0]), max_depth=int(args.para[1]),
				     min_samples_split=1, random_state=1)
    #print clf
if args.classifiers[0] == 'nb':
    clf = MultinomialNB()
    #print clf

clf.fit(x_train, y_train)
y_label = clf.predict(x_test)
#print type(y_label)
#print type(y_test)
#print y_label


################################################################################################################################
# results summary
y1_label = [] 
for i in range (len(y_label)):
	y1_label.append( clf.classes_.tolist().index(str(y_label[i])))

rs = dict()
f1 = dict()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(clf.classes_)):
	fpr[i], tpr[i], _ = roc_curve(y1_test, y1_label, pos_label=i)
	roc_auc[i] = auc(fpr[i], tpr[i])
	rs[i]=recall_score(y1_test, y1_label,average='macro',  pos_label=i)  
	f1[i]=f1_score(y1_test, y1_label, average='macro', pos_label=i)  
'''
print '\tfpr\ttpr\tauc\trecall\tf1 '
for i in range(len(clf.classes_)):
	print '{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(clf.classes_[i],fpr[i].mean(),tpr[i].mean(), roc_auc[i], rs[i],f1[i])
'''

count = 0 
for i in range (len(y_label)):
	if y_label[i] == y_test[i]:
		count += 1
d = float(count)/float(len(y_label))

################################################################################################################################
# write results to file

p = np.zeros((len(y_label), len(clf.classes_)), dtype = '|S8')
for i in range(len(x_test)):
	for j in range(len(clf.classes_)):
		prob = clf.predict_proba(x_test[i])
		p[i][j] = str("{:.5f}".format(prob[0][j]))

index = np.arange(len(y_label))
c = np.column_stack((index, y_label, p))
temp = ""
for row in c:
    for i in range(len(c[0])-1):
	temp += str(row[i]) + ","
    temp += str(row[len(c[0])-1])+"\n"
    #print temp
with open(args.out_file[0],"w") as f:
	f.write("@Index\n")
	f.write("@Predicted label\n")
	for i in range(len(clf.classes_)):
		f.write("@Prob of label: " + str(clf.classes_[i])+ "\n")
	f.write("\n")				
	f.write(temp)
	f.write("\n")
	f.write("=========================Summary================================\n")
	f.write("\n")
	f.write("Correctly Classified Instances :\t%d\t%0.4f%%\n" %(count,d*100))
	f.write("Incorrectly Classified Instances :\t%d\t%0.4f%%\n" %(len(y_label) - count,(1-d)*100))
	f.write("Total number of Instances :\t\t%d\n" %(len(y_label)))
	f.write("\n")
	f.write("=========================Detail=================================\n")
	f.write('\tfpr\ttpr\tauc\trecall\tf1\n ')
	for i in range(len(clf.classes_)):
		f.write('{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(clf.classes_[i],fpr[i].mean(),tpr[i].mean(), roc_auc[i], rs[i],f1[i]))
	f.write("\n")
	f.write("\n")
f.close()







