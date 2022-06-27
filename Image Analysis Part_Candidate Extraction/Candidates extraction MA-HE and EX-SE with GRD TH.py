import cv2 as cv
import numpy as np
import os
import functions as ft

def extract_bv_5method(fundus,num):
	# Green channel best contrast, we will use that one
	b, green_fundus, r = cv.split(fundus)
	cv.imwrite(f"images/results/presentation/1-green fundus_{num}.png", green_fundus)

	# Improve contrast
	clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
	contrast_enhanced_green_fundus = clahe.apply(green_fundus)
	cv.imwrite(f"images/results/presentation/2-CLAHE green fundus_{num}.png", contrast_enhanced_green_fundus)
	#ft.imshow_ft("green clahe", contrast_enhanced_green_fundus)

	# Denoise image - No significance difference in the order, CLAHE-NLMeans or NLMeans-CLAHE
	hh = 8
	window = 21
	templ = 7
	contrast_enhanced_green_fundus = cv.fastNlMeansDenoising(contrast_enhanced_green_fundus, h=hh,
															 searchWindowSize=window, templateWindowSize=templ)
	cv.imwrite(f"images/results/presentation/3-Denoise NLMeans green fundus_{num}.png", contrast_enhanced_green_fundus)
	#ft.imshow_ft("green clahe denoise", contrast_enhanced_green_fundus)


	# Invert image for better visualization of BV
	invert_im=ft.invert_image(contrast_enhanced_green_fundus)
	ft.imshow_ft("Green channel clahe invert", invert_im)
	cv.imwrite(f"images/results/presentation/4-Invert image_{num}.png", invert_im)

	# Remove background - Better contrast for BV
	background = cv.medianBlur(invert_im, 113)
	cv.imwrite(f"images/results/presentation/5-Background image_{num}.png", background)

	invert_im_without_back=cv.subtract(invert_im,background)
	cv.imwrite(f"images/results/presentation/6-invert_im_without_back_{num}.png", invert_im_without_back)
	#ft.imshow_ft("Background: median filter kernel", invert_im_without_back)

	# Not necessary to blur
	#invert_im_without_back = cv.medianBlur(invert_im_without_back, 7)
	#ft.imshow_ft("invert_im_without_back", invert_im_without_back)

	# Eliminate big lesions, we do not want to confuse them as vessels
	prep_TH = cv.morphologyEx(invert_im_without_back, cv.MORPH_TOPHAT,
							  cv.getStructuringElement(cv.MORPH_ELLIPSE, (51, 51)), iterations=1)
	cv.imwrite(f"images/results/presentation/7-Top hat big element_{num}.png", prep_TH)
	#ft.imshow_ft("TH", prep_TH)

	# Long and thin segments now (vessels), leave aside small objects
	prep_TH_OPEN = ft.max_operation_se_different_directions(prep_TH, cv.MORPH_OPEN, 101, 1, 20)
	cv.imwrite(f"images/results/presentation/8-Max open line different directions (seeds)_{num}.png", prep_TH_OPEN)
	#ft.imshow_ft("prep_TH_OPEN", prep_TH_OPEN)

	# We have seeds that are vessels for sure, reconstruct the vessels
	prep_TH_OPEN_recons = ft.reconstruction_dilations(prep_TH_OPEN, prep_TH, cv.MORPH_RECT, (3, 3))
	cv.imwrite(f"images/results/presentation/9- Reconstruct vessels form seed_{num}.png", prep_TH_OPEN_recons)
	#ft.imshow_ft("prep_TH_OPEN_recons", prep_TH_OPEN_recons)
	#cv.imwrite("46_bv.png", prep_TH_OPEN_recons)

	# Theshold BV with OTSU
	ret, prep_TH_OPEN_recons_thres = cv.threshold(prep_TH_OPEN_recons, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
	cv.imwrite(f"images/results/presentation/10- BV threshold OTSU_{num}.png", prep_TH_OPEN_recons_thres)
	#ft.imshow_ft("BV", prep_TH_OPEN_recons_thres)

	# Dilate to cover the borders of the vessels too
	prep_TH_OPEN_recons_thres_close = cv.morphologyEx(prep_TH_OPEN_recons_thres, cv.MORPH_DILATE,
													  cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)), iterations=1)
	cv.imwrite(f"images/results/presentation/11- Dilate BV threshold OTSU_{num}.png", prep_TH_OPEN_recons_thres_close)
	#ft.imshow_ft("BV-dilate", prep_TH_OPEN_recons_thres_close)

	# painting part, to make the "disappear"
	without_bv = cv.inpaint(invert_im, prep_TH_OPEN_recons_thres_close, 11, cv.INPAINT_TELEA)
	cv.imwrite(f"images/results/presentation/12- Fundus without BV_{num}.png", without_bv)
	#ft.imshow_ft("without hairs", without_bv)
	cv.imwrite(f"without_BV_{num}.png", without_bv)
	cv.imwrite(f"invert_{num}.png", invert_im)
	cv.imwrite(f"mask_bv_{num}.png", prep_TH_OPEN_recons_thres_close)

	return invert_im,prep_TH_OPEN_recons_thres_close,without_bv


def candidates_extraction(fundus,grd_MA,grd_HE,grd_EX,grd_SE,num,state):
	# Delete BloodVessels from the image
	if state=="not extracted bv":
		invert_clahe_denoise, mask_bv, without_bv = extract_bv_5method(fundus,num)
	# Read the images generated in the previous function
	else:
		invert_clahe_denoise = cv.imread(f"invert_{num}.png", cv.IMREAD_UNCHANGED)
		without_bv = cv.imread(f"without_BV_{num}.png", cv.IMREAD_UNCHANGED)
		mask_bv = cv.imread(f"mask_bv_{num}.png", cv.IMREAD_UNCHANGED)

	#Get mask of fundus, to diffentiate it from the borders in the image (255 inside and 0 outside)
	mask = ft.focus_circle_mask(invert_clahe_denoise)
	cv.imwrite(f"images/results/presentation/13- Mask of fundus_{num}.png", mask)

	# We need multiscale approach, we will divide into two pipelines, one for small lessions and one for big ones
	# Extract lesions with median blur, loss big lessions
	# Use mean so the borders will be less affected by color difference with the median filter
	mean = np.mean(without_bv[mask == 255])
	invert_im_gray = without_bv
	invert_im_gray[mask == 0] = mean
	ft.imshow_ft("invert_im_gray", invert_im_gray)
	background = cv.medianBlur(invert_im_gray, 153)
	ft.imshow_ft("Background: median filter kernel", background)
	background_16 = np.int16(background)
	prep_img_16 = invert_im_gray - background_16
	prep_img = np.uint8(cv.normalize(prep_img_16, 0, 255, norm_type=cv.NORM_MINMAX))
	ft.imshow_ft("Preprocess image: without noise and background", prep_img)
	cv.imwrite(f"images/results/presentation/14-S Remove background median filter and normalize_{num}.png", prep_img)

	# Extract small lesions with mean value
	# ma_he_small=cv.subtract(without_bv,background)
	# exudates_small=cv.subtract(background,without_bv)
	mean = np.mean(prep_img[mask == 255])
	ma_he_small = cv.subtract(prep_img, mean)
	exudates_small = cv.subtract(mean, prep_img)

	# Extract big lesions with mean value
	mean = np.mean(without_bv[mask == 255])
	ma_he_big = cv.subtract(without_bv, mean)
	exudates_big = cv.subtract(mean, without_bv)

	# Remove things that are not lesions (Bv and outside fundus)
	ma_he_small[mask_bv == 255] = 0
	ma_he_small[mask == 0] = 0
	ma_he_big[mask_bv == 255] = 0
	ma_he_big[mask == 0] = 0
	exudates_small[mask_bv == 255] = 0
	exudates_small[mask == 0] = 0
	exudates_big[mask_bv == 255] = 0
	exudates_big[mask == 0] = 0
	ft.imshow_ft("ma_he_small", ma_he_small)
	ft.imshow_ft("ma_he_big", ma_he_big)
	#cv.imwrite(f"ma_he_big_{num}.png", ma_he_big)

	# Remove background
	mean_mahe_big = np.mean(ma_he_big[mask == 255])
	sd_mahe_big = np.std(ma_he_big[mask == 255])
	#print(mean_mahe_big)
	#print(sd_mahe_big)
	ma_he_big_th = cv.subtract(ma_he_big, mean_mahe_big + 1.5 * sd_mahe_big)
	ft.imshow_ft("ma_he_big_th", ma_he_big_th)

	# CANNY does not work, low contrast
	# threshold=50
	# canny_output = cv.Canny(ma_he_big,threshold , threshold * 2)
	# # Find contours
	# # contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, mode=cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	# contours, hierarchy = cv.findContours(canny_output,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
	# print(len(contours))
	# # Draw contours
	# ma_he_big_copy=ma_he_big.copy()
	# #cv.drawContours(ma_he_big_copy, contours, -1, (255), 15, cv.LINE_8, hierarchy, 0)
	# cv.drawContours(ma_he_big_copy,contours,-1,(255),9)
	# # Show in a window
	# ft.imshow_ft('Contours', ma_he_big_copy)
	# cv.imwrite(f"ma_he_big_contours_{num}.png", ma_he_big_copy)

	# cv.imwrite(f"cand_ma_he_{num}.png", ma_he)

	ft.imshow_ft("exudates_small", exudates_small)
	ft.imshow_ft("exudates_big", exudates_big)
	#cv.imwrite(f"exudates_big_{num}.png", exudates_big)

	mean_exudates_big = np.mean(exudates_big[mask == 255])
	sd_exudates_big = np.std(exudates_big[mask == 255])
	# print(mean_exudates_big)
	# print(sd_exudates_big)
	exudates_big_th = cv.subtract(exudates_big, mean_exudates_big + 1.5 * sd_exudates_big)
	ft.imshow_ft("exudates_big_th", exudates_big_th)

	# Write 4 initial candidates
	cv.imwrite(f"images/results/presentation/15-S MA-HE Substract mean value_{num}.png", ma_he_small)
	cv.imwrite(f"images/results/presentation/16-S Exudates Substract mean value_{num}.png", exudates_small)
	cv.imwrite(f"images/results/presentation/17-B MA-HE Substract mean value_{num}.png", ma_he_big_th)
	cv.imwrite(f"images/results/presentation/18-B Exudates Substract mean value_{num}.png", exudates_big_th)

	# Improve candidate segmentation

	# ---------------      MA and HE    ----------------------------------------

	# min of TOP HAt dif directions to highlight MA
	mahe_MA = ft.min_operation_se_different_directions(ma_he_small, cv.MORPH_TOPHAT, 51, 1, 20)  # ,norm=1)
	ft.imshow_ft("mahe_MA", mahe_MA)
	ma_he_sum = cv.addWeighted(mahe_MA, 0.5, ma_he_small, 0.5, 0)
	ft.imshow_ft("ma_he_sum", ma_he_sum)

	# Morphology to delete small candidates and vessels
	ma_he_sum = cv.morphologyEx(ma_he_sum, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
	ft.imshow_ft("ma_he_sum after morphology", ma_he_sum)
	cv.imwrite(f"images/results/presentation/19-S MA-HE with highlighted MA and very thin vessels remove_{num}.png", ma_he_sum)

	# Just to compare results, not use in the final method
	ret, ma_he_binary_OTSU = cv.threshold(ma_he_sum, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
	ft.imshow_ft("OTSU", ma_he_binary_OTSU)

	# thresholds with mean and st
	mean = np.mean(ma_he_sum[mask == 255])
	st = np.std(ma_he_sum[mask == 255])
	# TL and TH inspired in histeresis thresholding as use by Canny
	ret, ma_he_TL = cv.threshold(ma_he_sum, mean + 1.5 * st, 255, cv.THRESH_BINARY)
	ft.imshow_ft("TL-result_ma_he_binary", ma_he_TL)
	ma_he_TL = cv.morphologyEx(ma_he_TL, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
	cv.imwrite(f"images/results/presentation/20-S MA-HE TL thresholding and small morph closing_{num}.png",ma_he_TL)

	ret, ma_he_TH = cv.threshold(ma_he_sum, mean + 3 * st, 255, cv.THRESH_BINARY)
	ft.imshow_ft("TH-result_ma_he_binary", ma_he_TH)
	cv.imwrite(f"images/results/presentation/21-S MA-HE TH thresholding seeds_{num}.png", ma_he_TH)

	result_ma_he_small = ft.reconstruction_dilations(ma_he_TH, ma_he_TL, cv.MORPH_RECT, (3, 3))
	ft.imshow_ft("TH-TL reconstruction small", result_ma_he_small)
	cv.imwrite(f"images/results/presentation/22-S MA-HE result- reconstruction TH to TL_{num}.png", result_ma_he_small)

	result_ma_he_big = ft.reconstruction_dilations(ma_he_TH, ma_he_big_th, cv.MORPH_RECT, (3, 3))
	ft.imshow_ft("TH-TL reconstruction big", result_ma_he_big)
	cv.imwrite(f"images/results/presentation/23-B MA-HE reconstruction TH to ma_he_big_{num}.png", result_ma_he_big)

	# borders does not matter, as they are in the other pipeline (small because of contrast with background median filter)
	result_ma_he_big = cv.medianBlur(result_ma_he_big, 11)
	result_ma_he_big = cv.morphologyEx(result_ma_he_big, cv.MORPH_OPEN,
									   cv.getStructuringElement(cv.MORPH_ELLIPSE, (51, 51)), iterations=1)
	# result_ma_he_big = cv.morphologyEx(result_ma_he_big, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)), iterations=1)
	ret, result_ma_he_big = cv.threshold(result_ma_he_big, mean + 1.5 * st, 255, cv.THRESH_BINARY)
	ft.imshow_ft("TL-result_ma_he_binary big", result_ma_he_big)
	cv.imwrite(f"images/results/presentation/24-B MA-HE result- median blur, opening, threshold mean_sd _{num}.png", result_ma_he_big)

	result_ma_he = cv.bitwise_or(result_ma_he_big, result_ma_he_small)
	cv.imwrite(f"images/results/cand_ma_he_{num}.png", result_ma_he)
	cv.imwrite(f"images/results/presentation/25- MA-HE final result-bitwise or _{num}.png",result_ma_he)

	# final version, inspired in Canny hysteresis thresholding, mixture of two approaches for big and small lesions
	ft.sensitivity_area(result_ma_he, grd_MA, "microan")
	ft.sensitivity_area(result_ma_he, grd_HE, "hemo")

	superp = ft.add_grdth(result_ma_he, grd_MA)
	superp = ft.add_grdth(superp, grd_HE)
	ft.imshow_ft("TL-TH final_result_MA_HE", superp)
	cv.imwrite(f"images/results/presentation/26- MA-HE GRD TH final result_{num}.png", superp)

	# For comparison only:
	# Only one TH, low threshold, many candidates, only small lesions
	ft.sensitivity_area(ma_he_TL, grd_MA, "microan")
	ft.sensitivity_area(ma_he_TL, grd_HE, "hemo")

	superp = ft.add_grdth(ma_he_TL, grd_MA)
	superp = ft.add_grdth(superp, grd_HE)
	ft.imshow_ft("2prueba-final_result_MA_HE", superp)

	# OTSU - too many candidates difficult to separate them,only small lesions
	ma_he_binary_OTSU
	ft.sensitivity_area(ma_he_binary_OTSU, grd_MA, "microan")
	ft.sensitivity_area(ma_he_binary_OTSU, grd_HE, "hemo")

	superp = ft.add_grdth(ma_he_binary_OTSU, grd_MA)
	superp = ft.add_grdth(superp, grd_HE)
	ft.imshow_ft("OTSU-final_result_MA_HE", superp)

	# ---------------      EX and SE    ----------------------------------------

	# min of TOP HAt dif directions to highlight small EX
	exudates_small_EX = ft.min_operation_se_different_directions(exudates_small, cv.MORPH_TOPHAT, 51, 1, 20)  # ,norm=1)
	ft.imshow_ft("exudates_small_EX", exudates_small_EX)
	exudates_sum = cv.addWeighted(exudates_small_EX, 0.2, exudates_small, 0.8, 0)
	ft.imshow_ft("exudates_small_EX_sum", exudates_sum)

	# morphology to delete too small candidates (noise likely)
	exudates_sum = cv.morphologyEx(exudates_sum, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
	ft.imshow_ft("exudates_sum after morphology", exudates_sum)
	cv.imwrite(f"images/results/presentation/27-S Exudates more importance to small EX and little opening_{num}.png", exudates_sum)

	# OTSU not use in final version, just for comparism
	ret, exudates_binary_OTSU = cv.threshold(exudates_sum, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
	ft.imshow_ft("OTSU exudates_sum", exudates_binary_OTSU)

	# thresholds with mean and st
	mean = np.mean(exudates_sum[mask == 255])
	st = np.std(exudates_sum[mask == 255])
	ret, exudates_TL = cv.threshold(exudates_sum, mean + 1 * st, 255, cv.THRESH_BINARY)
	ft.imshow_ft("TL-result_exudates_binary", exudates_TL)
	# exudates_TL=cv.morphologyEx(exudates_TL, cv.MORPH_CLOSE,cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5)))
	cv.imwrite(f"images/results/presentation/28-S Exudates TL thresholding mean_sd_{num}.png",exudates_TL)

	ret, exudates_TH = cv.threshold(exudates_sum, mean + 2.5 * st, 255, cv.THRESH_BINARY)
	ft.imshow_ft("TH-result_exudates_binary", exudates_TH)
	cv.imwrite(f"images/results/presentation/29-S Exudates TH thresholding mean_sd_{num}.png", exudates_TH)

	result_exudates_small = ft.reconstruction_dilations(exudates_TH, exudates_TL, cv.MORPH_RECT, (3, 3))
	ft.imshow_ft("TH-TL exudates reconstruction small", result_exudates_small)
	cv.imwrite(f"images/results/presentation/30-S Exudates result reconstruction_{num}.png", result_exudates_small)

	result_exudates_big = ft.reconstruction_dilations(exudates_TH, exudates_big_th, cv.MORPH_RECT, (3, 3))
	ft.imshow_ft("TH-TL exudates reconstruction big", result_exudates_big)
	cv.imwrite(f"images/results/presentation/31-B Exudates reconstruction big exudates_{num}.png",
			   result_exudates_big)

	# borders does not matter, as they are in the other pipeline
	result_exudates_big = cv.medianBlur(result_exudates_big, 3)
	result_exudates_big = cv.morphologyEx(result_exudates_big, cv.MORPH_OPEN,
										  cv.getStructuringElement(cv.MORPH_ELLIPSE, (51, 51)), iterations=1)
	# Erode to separate candidates, we do not care about borders as they are in the small pipeline
	result_exudates_big = cv.morphologyEx(result_exudates_big, cv.MORPH_ERODE,
										  cv.getStructuringElement(cv.MORPH_ELLIPSE, (21, 21)), iterations=1)
	ret, result_exudates_big = cv.threshold(result_exudates_big, mean + 1.5 * st, 255, cv.THRESH_BINARY)
	ft.imshow_ft("Result_exudates_binary big", result_exudates_big)
	cv.imwrite(f"images/results/presentation/32-B Exudates result- blur, open,erode,thresholding_{num}.png",  result_exudates_big)

	result_exudates = cv.bitwise_or(result_exudates_big, result_exudates_small)
	cv.imwrite(f"images/results/cand_exudates_{num}.png", result_exudates)
	cv.imwrite(f"images/results/presentation/33- Exudates final result_{num}.png",
			   result_exudates)

	# final version, inspired in Canny hysteresis thresholding, mixture of two approaches for big and small lesions
	ft.sensitivity_area(result_exudates, grd_EX, "exudates")

	superp = ft.add_grdth(result_exudates, grd_EX)
	# superp=ft.add_grdth(superp,grd_SE)
	ft.imshow_ft("TH-TL-final_result_exudates", superp)
	cv.imwrite(f"images/results/presentation/34- Exudates GRD TH final result_{num}.png",superp)

	# Only for comparism
	# Only one TH, low threshold, many candidates, only small lesions
	ft.sensitivity_area(exudates_TL, grd_EX, "exudates")

	superp = ft.add_grdth(exudates_TL, grd_EX)
	# superp=ft.add_grdth(superp,grd_SE)
	ft.imshow_ft("TL-final_result_exudates", superp)

	# OTSU - too many candidates difficult to separate them,only small lesions
	ft.sensitivity_area(exudates_binary_OTSU, grd_EX, "exudates")
	# ft.sensitivity_area(exudates_binary_OTSU,grd_SE,"hemo")

	superp = ft.add_grdth(exudates_binary_OTSU, grd_EX)
	# superp=ft.add_grdth(superp,grd_SE)
	ft.imshow_ft("OTSU-final_result_exudates", superp)

	return result_ma_he, result_exudates

num="10"
fundus = cv.imread(f"images/use/IDRiD_{num}.jpg")
grd_EX=cv.imread(f"images/use/IDRiD_{num}_EX.tif") #red
grd_MA=cv.imread(f"images/use/IDRiD_{num}_MA.tif") #red
grd_HE=cv.imread(f"images/use/IDRiD_{num}_HE.tif")#blue
grd_HE[np.where((grd_HE!=[0,0,0]).any(axis=2))] = [255,0,0]
grd_SE=0
#state="not extracted bv"
state="extracted bv"
result_ma_he, result_exudates = candidates_extraction(fundus,grd_MA,grd_HE,grd_EX,grd_SE,num,state)

cv.waitKey(0)
