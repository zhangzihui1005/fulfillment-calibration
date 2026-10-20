# Data access (not included)

Please obtain original datasets directly from their publishers. Raw files and order-level predictions are intentionally excluded.

- **Olist**: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce  
  License reported by publisher: **CC BY-NC-SA 4.0**. Download original nine CSVs; save under `data/raw/olist/` or pass `--olist-zip /absolute/path/Olist.zip`.
- **DataCo Smart Supply Chain**: https://data.mendeley.com/datasets/8gx2fvg2k6/5  
  Check the exact record/version and redistribution terms before use.
- **LaDe-P**: https://huggingface.co/datasets/Cainiao-AI/LaDe-P  
  Dataset page lists **Apache-2.0**; source repo and data citation: https://github.com/wenhaomin/LaDe.

Do not commit personally identifying fields, credentials, raw customer/order records, GPS traces, or order-level outputs. Attribution and repository-code licensing are independent of original-data licensing.

Note: DataCo shipping-risk labels and LaDe-P pickup deadlines are not equivalent to Olist final-delivery delay labels.
