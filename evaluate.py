from cmath import nan
import os
import cv2
import glob
import numpy as np
import math
import torch

# import time
# import scipy.ndimage
# import tifffile
# from PIL import Image

def load_label(label_path):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if len(label.shape) == 3:
        label = label[:,:,0]
    for k, v in label_mapping.items():
        label[label == k] = v
    return label


def save_visible(file_path, label):
    # print(file_path, label.shape)
    row = label.shape[0]
    col = label.shape[1]
    image = np.ones((row, col, 3)) * visible_mapping[-1]
    image = np.uint8(image)

    for k, v in visible_mapping.items():
        image[label == k] = v
    
    # image[label == 1] = [34, 47, 157]
    # image[label == 2] = [175, 88, 144]
    # image[label == 3] = [112, 53, 168]
    # image[label == 4] = [122, 200, 177]
    # image[label == 5] = [82, 126, 179]
    # image[label == 6] = [52, 182, 199]
    # image[label == 7] = [34, 99, 30]
    # image[label == 8] = [140, 105, 42]
    image = np.uint8(image)
    cv2.imwrite(file_path, image)

def corr(A,B):
	################ with zero
    # a = A-A.mean()
    # b = B-B.mean()
    # return (a*b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))

	################ without zero
	A = np.where(B!=0,A,np.nan)
	B = np.where(B!=0,B,np.nan)
	a = A-np.nanmean(A)
	b = B-np.nanmean(B)
	if (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2))) == 0:
		return nan
	else: 
		return np.nansum(a*b) / (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2)))

def corr_wZero(A,B):
	################ with zero
    a = A-A.mean()
    b = B-B.mean()
    if (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())) ==0:
        return nan
    else:
    	return (a*b).sum() / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum()))
 
def corr_wCla(A,B,C):	
	A = np.where(C!=255,A,np.nan)
	B = np.where(C!=255,B,np.nan)	
	# A = np.where(B!=0,A,np.nan)
	# B = np.where(B!=0,B,np.nan)
	# B_nonnan_cnt = np.count_nonzero(~np.isnan(B))
	# B_nz_cnt = np.count_nonzero(B)
	a = A-np.nanmean(A)
	b = B-np.nanmean(B)
	if (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2))) == 0:
		return nan#, B_nonnan_cnt, B_nz_cnt
	else: 
		return np.nansum(a*b) / (np.sqrt(np.nansum(a**2)) * np.sqrt(np.nansum(b**2)))#, B_nonnan_cnt, B_nz_cnt
 
########################################################################################################################
	
def r_square(A,B):	
	A = np.where(B!=0,A,np.nan)
	B = np.where(B!=0,B,np.nan)
	if np.nansum((B-np.nanmean(B))**2) == 0:
		return nan
	else:
		return 1.0 - ( np.nansum(((B-A)**2))  / np.nansum((B-np.nanmean(B))**2) )

def r_square_wZero(A,B):	
	if np.nansum((B-np.nanmean(B))**2) ==0:
		return nan
	else:
		return 1.0 - (    ((B-A)**2).sum() / ((B-B.mean())**2).sum() )

def r_square_wCla(A,B,C):	
	A = np.where(C!=255,A,np.nan)
	B = np.where(C!=255,B,np.nan)
	# A = np.where(B!=0,A,np.nan)
	# B = np.where(B!=0,B,np.nan)
	if np.nansum((B-np.nanmean(B))**2) == 0:
		return nan
	else:
		return 1.0 - ( np.nansum(((B-A)**2))  / np.nansum((B-np.nanmean(B))**2) )	



# def calc_accuracy(gt_file, result_file, class_num=NUM_CLASSES):
#     print("gt:", gt_file, ", result:", result_file)
#     gt = load_label(gt_file)
#     result = load_label(result_file)

#     row1, col1 = gt.shape[:2]
#     row2, col2 = result.shape[:2]
#     row = min(row1, row2)
#     col = min(col1, col2)

#     result = result[:row, :col]
#     gt = gt[:row, :col]

#     # result[result > (class_num - 1)] = -1
#     # gt[gt > (class_num - 1)] = -1

#     count_table = np.zeros([class_num, class_num], dtype=np.int)
#     for x in range(class_num):
#         for y in range(class_num):
#             count_table[x][y] = np.sum(np.logical_and(gt == x, result == y))

#     return count_table


# def print_write_txt(script, fid):
# 	fid.write(script)
# 	fid.write('\n')
# 	print(script)


# def print_from_table(count_table, filename, fid, metric=''):
# 	print_write_txt("="* 30, fid)
# 	print_write_txt(f"FileName = {filename}", fid)

# 	iou_list = []
# 	for class_idx in range(1, NUM_CLASSES):
# 		tp = count_table[class_idx][class_idx]
# 		tp_fn = np.sum(count_table, axis=1)[class_idx]
# 		tp_fp = np.sum(count_table, axis=0)[class_idx]
# 		acc = round(tp/tp_fn, 4)
# 		iou = round(tp/(tp_fn + tp_fp - tp), 4)
# 		if metric =='iou':
# 			print_write_txt(f"class {class_idx} = {iou}", fid)
# 			if not math.isnan(iou):
# 				iou_list.append(iou)
# 		else:
# 			print_write_txt(f"class {class_idx} = {acc}", fid)

# 	total_px = np.sum(count_table)
# 	total_tp = np.sum(np.diag(count_table)[1:])
# 	total_tn = np.diag(count_table)[0]
# 	total_fn = np.sum(count_table[1:]) - total_tp
# 	total_fp = np.sum(count_table[0]) - total_tn
# 	total_acc = round(total_tp / (total_tp + total_fn), 4)
# 	print_write_txt(f"TP = {round(total_tp/total_px, 4)}", fid)
# 	print_write_txt(f"TN = {round(total_tn/total_px, 4)}", fid)
# 	print_write_txt(f"FN = {round(total_fn/total_px, 4)}", fid)
# 	print_write_txt(f"FP = {round(total_fp/total_px, 4)}", fid)
# 	if metric == 'iou':
# 		miou = round(np.mean(iou_list), 4)
# 		print_write_txt(f"mIou = {miou}", fid)
# 	else:
# 		#print_write_txt(f"Accuracy = {total_acc}", fid)
# 		print_write_txt(f"Correlation = {total_acc}", fid)


# def calc_result_excel(result_path, test_label_path, size, each_file_result=True, today=None):
# 	if size == 1024:
# 		image_endfix_len = 9
# 		label_endfix = "_FGT_1024"
# 	else :			
# 		image_endfix_len = 4
# 		#label_endfix = "_FGT"
# 		label_endfix = ""

#         #if test_label_path == "/data/Forest_Carbon/test/label" :
#         #        metri = iou
#         #else :
#         #        metri = ''

# 	result_txt = os.path.join(result_path, os.path.basename(result_path) + '.txt')
# 	f_txt = open(result_txt, 'at')
# 	result_files = glob.glob(result_path + "/*.tif")

# 	for result_file in result_files:
# 		#gt_file = os.path.join(test_label_path, os.path.basename(result_file))
# 		gt_file = os.path.join(test_label_path, os.path.basename(result_file)[:-image_endfix_len] + label_endfix + ".tif")
# 		if not os.path.exists(gt_file):
# 			print("Not exist", gt_file)
# 			continue
# 		count_table = calc_accuracy(gt_file, result_file, NUM_CLASSES)
# 		# print("*" * 20)

# 		excel_file = os.path.join(result_path, '%s_result.xlsx' % os.path.basename(result_file))
# 		if os.path.exists(excel_file):
# 			os.remove(excel_file)
# 		wb = Workbook()
# 		ws = wb.active

# 		for col in range(NUM_CLASSES):
# 			for row in range(NUM_CLASSES):
# 				ws.cell(row=row + 1, column=col + 1, value=count_table[row][col])

# 		iou_list = [ ]
# 		true_count = 0
# 		class_count = 0
# 		for class_idx in range(NUM_CLASSES):
# 			# count_table[0][class_idx] = 0
# 			tp = count_table[class_idx][class_idx]
# 			tp_fp = np.sum(count_table, axis=0)[class_idx]
# 			tp_fn = np.sum(count_table, axis=1)[class_idx]
# 			if tp_fp + tp_fn == 0:
# 				iou = 0
# 				acc = 'NaN'
# 			else:
# 				class_count += 1
# 				iou = tp / (tp_fp + tp_fn - tp)
# 				if tp_fn == 0:
# 					acc = 'NaN'
# 				else:
# 					acc = tp / tp_fn
# 			iou_list.append(iou)
# 			ws.cell(row=NUM_CLASSES + 1, column=class_idx + 1, value=acc)
# 			ws.cell(row=NUM_CLASSES + 2, column=class_idx + 1, value=iou)
# 			true_count += count_table[class_idx][class_idx]

# 		ws.cell(row=NUM_CLASSES + 1, column=1, value="Acc")
# 		ws.cell(row=NUM_CLASSES + 2, column=1, value="Iou")
# 		mIou = np.sum(iou_list[1:]) / (class_count - 1)
# 		fIou = np.sum(np.multiply(iou_list, np.sum(count_table[:,1:], axis=1))) / np.sum(count_table[:,1:])
# 		accuracy = true_count / np.sum(count_table) #round(true_count / np.sum(count_table), 4)
# 		ws.cell(row=NUM_CLASSES + 3, column=1, value="mIou")
# 		ws.cell(row=NUM_CLASSES + 3, column=2, value=mIou)
# 		ws.cell(row=NUM_CLASSES + 4, column=1, value="fIou")
# 		ws.cell(row=NUM_CLASSES + 4, column=2, value=fIou)
# 		ws.cell(row=NUM_CLASSES + 5, column=1, value="accuracy")
# 		ws.cell(row=NUM_CLASSES + 5, column=2, value=accuracy)

# 		wb.save(excel_file)

# 		#print_from_table(count_table, os.path.basename(gt_file), f_txt, 'iou')
# 		print_from_table(count_table, os.path.basename(gt_file), f_txt, metri)

# 	result_excel = os.path.join(result_path, os.path.basename(result_path) + '.xlsx')
# 	if os.path.exists(result_excel):
# 		os.remove(result_excel)
# 	excel_files = glob.glob(result_path + "/*.xlsx")
# 	print("result_excel file = ", result_excel)

# 	wb = Workbook()
# 	ws = wb.active
# 	if each_file_result:
# 		ws_f = wb.create_sheet('file')
# 		ws_f['A1'].value = "filename"
# 		ws_f['B1'].value = "Accuracy"
# 	count_table = np.zeros([NUM_CLASSES, NUM_CLASSES], dtype=np.int)
# 	for idx, excel_file in enumerate(excel_files):
# 		all_px = 0
# 		true_px = 0
# 		print(excel_file)
# 		f_wb = load_workbook(excel_file)
# 		f_ws = f_wb.worksheets[0]
# 		for i in range(NUM_CLASSES):
# 			for j in range(NUM_CLASSES):
# 				count_table[i][j] += int(f_ws.cell(row=i + 1, column=j + 1).value)
# 				#if i > 0 and j > 0:
# 				px = f_ws.cell(row=i + 1, column=j + 1).value
# 				all_px += px
# 				if i == j:
# 					true_px += px
# 		if each_file_result:
# 			tif_name = os.path.basename(excel_file).replace("_result.xlsx", "")
# 			ws_f.cell(row=idx + 2, column=1).value = tif_name
# 			if all_px != 0:
# 				ws_f.cell(row=idx + 2, column=2).value = true_px / all_px
# 	# print(count_table)

# 	for col in range(NUM_CLASSES):
# 		for row in range(NUM_CLASSES):
# 			ws.cell(row=row + 1, column=col + 1, value=count_table[row][col])

# 	iou_list = [ ]
# 	acc_list = [ ]
# 	true_count = 0
# 	class_count = 0
# 	for class_idx in range(NUM_CLASSES):
# 		tp = count_table[class_idx][class_idx]
# 		tp_fp = np.sum(count_table, axis=0)[class_idx]
# 		tp_fn = np.sum(count_table, axis=1)[class_idx]
# 		if tp_fp + tp_fn == 0:
# 			iou = 0
# 			acc = 0
# 		else:
# 			class_count += 1
# 			iou = tp / (tp_fp + tp_fn - tp)
# 			acc = tp / tp_fn
# 		iou_list.append(iou)
# 		acc_list.append(acc)
# 		ws.cell(row=NUM_CLASSES + 1, column=class_idx + 1, value=acc)
# 		ws.cell(row=NUM_CLASSES + 2, column=class_idx + 1, value=iou)
# 		true_count += count_table[class_idx][class_idx]

# 	mIou = np.sum(iou_list[1:]) / (class_count - 1)
# 	fIou = np.sum(np.multiply(iou_list, np.sum(count_table, axis=1))) / np.sum(count_table)
# 	accuracy = round(true_count / np.sum(count_table), 4)
# 	accuracy_target = round(np.sum(np.diag(count_table)[1:])/np.sum(count_table[1:]), 4)
# 	#ws.cell(row=NUM_CLASSES + 1, column=1, value="Acc")
# 	#ws.cell(row=NUM_CLASSES + 2, column=1, value="Iou")
# 	ws.cell(row=NUM_CLASSES + 3, column=1, value="mIou")
# 	ws.cell(row=NUM_CLASSES + 4, column=1, value="fIou")
# 	# ws.cell(row=NUM_CLASSES + 5, column=1, value="accuracy")
# 	ws.cell(row=NUM_CLASSES + 6, column=1, value="Overall Acc")
# 	ws.cell(row=NUM_CLASSES + 3, column=2, value=mIou)
# 	ws.cell(row=NUM_CLASSES + 4, column=2, value=fIou)
# 	# ws.cell(row=NUM_CLASSES + 5, column=2, value=accuracy)
# 	ws.cell(row=NUM_CLASSES + 6, column=2, value=accuracy_target)

# 	if today != None:
# 		ws.cell(row=NUM_CLASSES + 7, column=1, value="Time stamp")
# 		time_str = '{:02d}/{:02d}/{:02d} {:02d}:{:02d}'.format(today.tm_year, today.tm_mon, today.tm_mday, today.tm_hour, today.tm_min)
# 		ws.cell(row=NUM_CLASSES + 7, column=2, value=time_str)
# 		print(time_str)

# 	#print_from_table(count_table, "Result All", f_txt, 'iou')
# 	print_from_table(count_table, "Result All", f_txt, '')

# 	print(f'Acc List : {acc_list}')
# 	print(f'Iou List : {iou_list}')
# 	print(f'Result : {mIou}, {fIou}, {accuracy_target}')
# 	print(f'Result All : {mIou}, {fIou}, {accuracy_target}, {str(iou_list)[1:-1]}, {str(acc_list)[1:-1]}')

# 	wb.save(result_excel)
# 	f_txt.close()

# 	return accuracy


# def calc_result_area(result_path, area_str):
# 	if area_str == None:
# 		excel_files = glob.glob(result_path + "/*.xlsx")
# 	else:
# 		excel_files = glob.glob(result_path + f"/*{area_str}*.xlsx")

# 	count_table = np.zeros([NUM_CLASSES, NUM_CLASSES], dtype=np.int)
# 	for excel_file in excel_files:
# 		all_px = 0
# 		true_px = 0
# 		f_wb = load_workbook(excel_file)
# 		f_ws = f_wb.worksheets[0]
# 		for i in range(NUM_CLASSES):
# 			for j in range(NUM_CLASSES):
# 				count_table[i][j] += int(f_ws.cell(row=i + 1, column=j + 1).value)
# 				#if i > 0 and j > 0:
# 				px = f_ws.cell(row=i + 1, column=j + 1).value
# 				all_px += px
# 				if i == j:
# 					true_px += px

# 	iou_list = [ ]
# 	acc_list = [ ]
# 	true_count = 0
# 	class_count = 0
# 	for class_idx in range(NUM_CLASSES):
# 		tp = count_table[class_idx][class_idx]
# 		tp_fp = np.sum(count_table, axis=0)[class_idx]
# 		tp_fn = np.sum(count_table, axis=1)[class_idx]
# 		if tp_fp + tp_fn == 0:
# 			iou = 0
# 			acc = 0
# 		else:
# 			class_count += 1
# 			iou = tp / (tp_fp + tp_fn - tp)
# 			acc = tp / tp_fn
# 		iou_list.append(iou)
# 		acc_list.append(acc)
# 		true_count += count_table[class_idx][class_idx]

# 	mIou = np.sum(iou_list[1:]) / (class_count - 1)
# 	fIou = np.sum(np.multiply(iou_list, np.sum(count_table, axis=1))) / np.sum(count_table)
# 	accuracy = round(true_count / np.sum(count_table), 4)
# 	accuracy_target = round(np.sum(np.diag(count_table)[1:])/np.sum(count_table[1:]), 4)

# 	print("Area : ", area_str)
# 	print(f'Acc List : {acc_list}')
# 	print(f'Iou List : {iou_list}')
# 	print(f'Result : {mIou}, {fIou}, {accuracy_target}')
# 	# print(f'Result All : {mIou}, {fIou}, {accuracy_target}, {str(iou_list)[1:-1]}, {str(acc_list)[1:-1]}')


# if __name__ == "__main__":
# 	# calc_result_excel("/raid/dataset/aidata/211204/aerial_jj/test/result/20211205_1007", "/raid/dataset/aidata/211204/aerial_jj/test/label")
# 	calc_result_excel("/raid/dataset/forest/211207/forest_st/test/result/20211209_1753", "/raid/dataset/forest/211207/forest_st/test/label")
# 	calc_result_area("/raid/dataset/forest/211207/forest_st/test/result/20211209_1753", "GS")
# 	calc_result_area("/raid/dataset/forest/211207/forest_st/test/result/20211209_1753", "JL")
# 	calc_result_area("/raid/dataset/forest/211207/forest_st/test/result/20211209_1753", "JJ")
