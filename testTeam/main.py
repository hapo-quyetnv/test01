from tkinter import *
import tkinter
from tkinter.ttk import *
import pandas as pd
import numpy as np
root = Tk()

class TreeNode(object):
    def __init__(self, ids = None, children = [], entropy = 0, depth = 0):
        self.ids = ids           
        self.entropy = entropy   
        self.depth = depth       
        self.split_attribute = None # chua nhung thuoc tinh duoc chon ma khong phai nut la
        self.children = children # danh sach ca nut con cua ca nut duoc chon ben trn tuong ung
        self.order = None       # order of values of split_attribute in children (gia tri cua split_attribute nam trong children)
        self.label = None       # label of node if it is a leaf (nhan cua nut tuong ung neu la nut la)

    def set_properties(self, split_attribute, order):
        self.split_attribute = split_attribute
        self.order = order

    def set_label(self, label):
        self.label = label


def entropy(freq):
    # remove prob 0 
    freq_0 = freq[freq.nonzero()[0]]
    # print(freq)
    prob_0 = freq_0/float(freq_0.sum())
    return -np.sum(prob_0*np.log(prob_0))

class DecisionTreeID3(object):
    def __init__(self, max_depth):
        self.root = None
        self.max_depth = max_depth
        self.Ntrain = 0
    
    def fit(self, data, target):
        self.Ntrain = data.count()[0]
        self.data = data 
        self.attributes = list(data)
        self.target = target 
        self.labels = target.unique()
        ids = range(self.Ntrain)
        self.root = TreeNode(ids = ids, entropy = self._entropy(ids), depth = 0) # selt.root trở thành 1 instance của class TreeNote
        queue = [self.root]
        while queue:
            node = queue.pop()
            if node.depth < self.max_depth :
                node.children = self._split(node)
                if not node.children: #leaf node
                    self._set_label(node)
                queue += node.children
            else:
                self._set_label(node)
                
    def _entropy(self, ids):
        # calculate entropy of a node with index ids
        if len(Sơnids) == 0: return 0
        ids = [i+1 for i in ids] # panda series index starts from 1
        freq = np.array(self.target[ids].value_counts())
        # print(freq)
        return entropy(freq)

    def _set_label(self, node):
        # find label for a node if it is a leaf
        # simply chose by major voting 
        target_ids = [i + 1 for i in node.ids]  # target is a series variable
        node.set_label(self.target[target_ids].mode()[0]) # lay ra nhan chiem da so trong mang target ben trrn
    
    def _split(self, node):
        ids = node.ids
        best_gain = 0
        best_splits = []
        best_attribute = None
        order = None
        sub_data = self.data.iloc[ids, :]
        for i, att in enumerate(self.attributes): # biến các attr 'parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health' thành mảng với i là key các att là các attr
            values = self.data.iloc[ids, i].unique().tolist() # lấy các giá trị của attr, ở đây to list để chuyển sang kiểu list, ko có to list thì nó ở dạng numpy.array
            if len(values) == 1: continue # entropy = 0 khi mà attr chỉ có duy nhất 1 giá trị
            splits = []
            for val in values:
                # val ở đây là các giá trị cụ thể của thuộc tính. ví dụ thuộc tính pareants có các giá trị cụ thể là usual, pretentious, great_pret
                sub_ids = sub_data.index[sub_data[att] == val] #sub_ids ở đây là Các id mà thuộc tính đó có giá trị là val đó
                splits.append([sub_id-1 for sub_id in sub_ids]) # trong mảng nên đoạn này cho sub_id chạy từ 0, đầu ra splits là mảng lớn chứa các mảng con, mỗi mảng con chưa các id của giá trị thuộc tính tương ứng

            # information gain
            HxS= 0
            for split in splits:
                HxS += len(split)*self._entropy(split)/len(ids)
            gain = node.entropy - HxS
            if gain > best_gain:
                best_gain = gain
                best_splits = splits
                best_attribute = att
                order = values
        node.set_properties(best_attribute, order)
        child_nodes = [TreeNode(ids = split,
                     entropy = self._entropy(split), depth = node.depth + 1) for split in best_splits]
        return child_nodes # các thuộc tính được chọn

    def predict(self, new_data):
        npoints = new_data.count()[0]
        labels = [None]*npoints
        for n in range(npoints):
            x = new_data.iloc[n, :]
            node = self.root
            while node.children:
                node = node.children[node.order.index(x[node.split_attribute])]
            labels[n] = node.label
            
        return labels

    def predictTest(self, data):
        x = data
        node = self.root
        while node.children:
            node = node.children[node.order.index(x[node.split_attribute])]
        label = node.label
        return label
class Window(tkinter.Frame):
    def __init__(self, master = None):
        Frame.__init__(self,master)
        self.master = master
        self.init_window()
    def init_window(self):
        self.master.title("Demo thuật toán ID3")
        self.pack(fill=BOTH, expand=1)
        self.drawEncoding()
   
    def drawEncoding(self):
        self.encodeVar = StringVar()
        self.encodelabel = Label(root, textvariable=self.encodeVar,font=("Arial Bold", 36))
        self.encodelabel.place(x=450, y=10)
        self.encodeVar.set("Phiếu Thông Tin ")
        
        self.parent = StringVar()
        rad1 = Radiobutton(root,text='great_pret', value='great_pret', variable=self.parent)
        rad2 = Radiobutton(root,text='pretentious', value='pretentious', variable=self.parent)
        rad3 = Radiobutton(root,text='usual', value='usual', variable=self.parent)
        self.lengthText1 = StringVar()
        self.lengthlabel1 = Label(root, textvariable=self.lengthText1)
        self.lengthlabel1.place(x=10, y=100)
        self.lengthText1.set("Chọn parents: ")
        rad1.place(x=10,y=130)
        rad2.place(x=110,y=130)
        rad3.place(x=210,y=130)
        
        self.has_nurs  = StringVar()
        rad4 = Radiobutton(root,text='proper', value='proper', variable=self.has_nurs )
        rad5 = Radiobutton(root,text='less_proper', value='less_proper', variable=self.has_nurs )
        rad6 = Radiobutton(root,text='improper', value='improper', variable=self.has_nurs )
        rad7 = Radiobutton(root,text='critical', value='critical', variable=self.has_nurs )
        rad8 = Radiobutton(root,text='very_crit', value='very_crit', variable=self.has_nurs )
        self.lengthText2 = StringVar()
        self.lengthlabel2 = Label(root, textvariable=self.lengthText2)
        self.lengthlabel2.place(x=10, y=170)
        self.lengthText2.set("Chọn has_nurs: ")
        rad4.place(x=10,y=190)
        rad5.place(x=110,y=190)
        rad6.place(x=210,y=190)
        rad7.place(x=310,y=190)
        rad8.place(x=410,y=190)
        
        
        self.form = StringVar()
        rad9 = Radiobutton(root,text='complete', value='complete', variable=self.form)
        rad10 = Radiobutton(root,text='completed', value='completed', variable=self.form)
        rad11 = Radiobutton(root,text='incomplete', value='incomplete', variable=self.form)
        rad12 = Radiobutton(root,text='foster', value='foster', variable=self.form)
        self.lengthText3 = StringVar()
        self.lengthlabel3 = Label(root, textvariable=self.lengthText3)
        self.lengthlabel3.place(x=10, y=240)
        self.lengthText3.set("Chọn form: ")
        rad9.place(x=10,y=270)
        rad10.place(x=110,y=270)
        rad11.place(x=210,y=270)
        rad12.place(x=310,y=270)
        
        self.childrens  = StringVar()
        rad13 = Radiobutton(root,text='1', value='1', variable=self.childrens)
        rad14 = Radiobutton(root,text='2', value='2', variable=self.childrens)
        rad15 = Radiobutton(root,text='3', value='3', variable=self.childrens)
        rad16 = Radiobutton(root,text='more', value='more', variable=self.childrens)
        self.lengthText4 = StringVar()
        self.lengthlabel4 = Label(root, textvariable=self.lengthText4)
        self.lengthlabel4.place(x=10, y=300)
        self.lengthText4.set("Chọn children: ")
        rad13.place(x=10,y=330)
        rad14.place(x=110,y=330)
        rad15.place(x=210,y=330)
        rad16.place(x=310,y=330)
        
        self.housing  = StringVar()
        rad17 = Radiobutton(root,text='convenient', value='convenient', variable=self.housing)
        rad18 = Radiobutton(root,text='critical', value='critical', variable=self.housing)
        rad19 = Radiobutton(root,text='less_conv', value='less_conv', variable=self.housing)
        self.lengthText5 = StringVar()
        self.lengthlabel5 = Label(root, textvariable=self.lengthText5)
        self.lengthlabel5.place(x=10, y=360)
        self.lengthText5.set("Chọn housing: ")
        rad17.place(x=10,y=390)
        rad18.place(x=110,y=390)
        rad19.place(x=210,y=390)
        
        self.finance  = StringVar()
        rad20 = Radiobutton(root,text='convenient', value='convenient', variable=self.finance)
        rad21 = Radiobutton(root,text='inconv', value='inconv', variable=self.finance)
        self.lengthText6 = StringVar()
        self.lengthlabel6 = Label(root, textvariable=self.lengthText6)
        self.lengthlabel6.place(x=10, y=420)
        self.lengthText6.set("Chọn finance: ")
        rad20.place(x=10,y=450)
        rad21.place(x=110,y=450)
        
        
        self.social  = StringVar()
        rad22 = Radiobutton(root,text='problematic', value='problematic', variable=self.social)
        rad23 = Radiobutton(root,text='nonprob', value='nonprob', variable=self.social)
        rad24 = Radiobutton(root,text='slightly_prob', value='slightly_prob', variable=self.social)
        self.lengthText7 = StringVar()
        self.lengthlabel7 = Label(root, textvariable=self.lengthText7)
        self.lengthlabel7.place(x=10, y=480)
        self.lengthText7.set("Chọn social: ")
        rad22.place(x=10,y=510)
        rad23.place(x=110,y=510)
        rad24.place(x=210,y=510)
        
        self.health  = StringVar()
        rad25 = Radiobutton(root,text='recommended', value='recommended', variable=self.health)
        rad26 = Radiobutton(root,text='priority', value='priority', variable=self.health)
        rad27 = Radiobutton(root,text='not_recom', value='not_recom', variable=self.health)
        self.lengthText8 = StringVar()
        self.lengthlabel8 = Label(root, textvariable=self.lengthText8)
        self.lengthlabel8.place(x=10, y=540)
        self.lengthText8.set("Chọn health: ")
        rad25.place(x=10,y=570)
        rad26.place(x=110,y=570)
        rad27.place(x=210,y=570)

        self.encodeButton = Button(self, text="Đánh Giá", command=self.run)
        self.encodeButton.place(x=100, y=630)
        
        self.boxVar7 = StringVar()
        self.boxlabel7 = Label(root, textvariable=self.boxVar7,font=(None, 17))
        self.boxlabel7.place(x=700, y=100)
        self.boxVar7.set("Nhãn")
        # Text ghi văn bản cần giấu
        self.labelResult = StringVar()
        self.labelResultlabel2 = Label(root, textvariable=self.labelResult, font=(None, 20))
        self.labelResultlabel2.place(x=700, y=150)
        
        self.boxVar8 = StringVar()
        self.boxlabel8 = Label(root, textvariable=self.boxVar8,font=(None, 17))
        self.boxlabel8.place(x=700, y=250)
        self.boxVar8.set("Độ Chính Xác")
        self.testResult = StringVar()
        self.testResultlabel2 = Label(root, textvariable=self.testResult, font=(None, 20))
        self.testResultlabel2.place(x=700, y=300)
        
    def run(self):
        df = pd.read_csv('trainN.csv', index_col = 0)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        tree = DecisionTreeID3(max_depth = 3)
        tree.fit(X, y)
        dftest = pd.read_csv('testN.csv', index_col= 0)
        Xtest = dftest.iloc[:, :-1]
        yResultTest = tree.predict(Xtest)
        listNhan = list(dftest.iloc[:, 8])
        count = 0
        for i in range(len(listNhan)) :
            if yResultTest[i] == listNhan[i] : count += 1
        testRs = (count / len(listNhan)) * 100
        self.testResult.set(str(testRs) + ' %')
        data = {'parents': self.parent.get(), 'has_nurs': self.has_nurs.get(), 'form': self.form.get(), 'childrens': self.childrens.get(), 'housing': self.housing.get(), 'finance': self.finance.get(), 'social': self.social.get(), 'health': self.health.get()}
        result = tree.predictTest(data)
        self.labelResult.set(result)
        print(self.parent.get())
root.geometry("1200x700")
app = Window(root)
root.mainloop()