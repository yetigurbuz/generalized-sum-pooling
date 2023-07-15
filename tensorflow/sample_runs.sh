""" BNInception Runs """

""" Online Products our Method """
python trainModel.py --device=0 --repeat=3 --dataset=SOP --cfg=dml_large --loss=zsr/xbm/original_contrastive --pooling=GeneralizedSumPooling  


""" InShop our Method """
python trainModel.py --device=0 --repeat=3 --dataset=InShop --cfg=dml_large --loss=zsr/xbm/original_contrastive --pooling=GeneralizedSumPooling 


""" Cars our Method """
python trainModel.py --device=0 --repeat=3 --dataset=Cars196 --cfg=dml_small --loss=zsr/xbm/original_contrastive --pooling=GeneralizedSumPooling

""" Cub our Method """
python trainModel.py --device=0 --repeat=3 --dataset=CUB200_2011 --cfg=dml_small --loss=zsr/xbm/original_contrastive --pooling=GeneralizedSumPooling
