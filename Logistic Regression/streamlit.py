
import pandas as pd
import numpy as np
import streamlit as st 
from sqlalchemy import create_engine
from urllib.parse import quote
import joblib, pickle
from statsmodels.regression.linear_model import OLSResults

model = OLSResults.load("logit_model.pkl")
imp_enc_scale = joblib.load('imp_enc_scale')
winsor = joblib.load('winsor')


def predict(data, user, pw, db):
    
    engine = create_engine(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))
    
    clean = pd.DataFrame(imp_enc_scale.transform(data), columns = imp_enc_scale.get_feature_names_out())
    clean[list(clean.iloc[:,0:4].columns)] = winsor.transform(clean[list(clean.iloc[:,0:4].columns)])

    
    
    prediction = model.predict(clean)
    
    optimal_threshold = 0.41346512057627866
    data["Clicked_on_Ad"] = np.zeros(len(prediction))
    
    #taking threshold value and above the prob value will be treated as 
    
    data.loc[prediction > optimal_threshold, "Clicked_on_Ad"] = 1
    data[['Clicked_on_Ad']] = data[['Clicked_on_Ad']].astype('int64')
    
    data.to_sql('clicked_on_ad_predictions', con = engine, if_exists = 'replace', )
    
    return data

def main():
    
    st.title("Clicked_on_Ad prediction")
    st.sidebar.title("Clicked_on_Ad prediction")

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Clicked_on_Adprediction </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file" ,type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    if uploadedFile is not None :
        try:

            data=pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("Upload the new data using CSV or Excel file.")
    
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here", type = 'password')
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    
    if st.button("Predict"):
        result = predict(data, user, pw, db)
        
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap=True)
        st.table(result.style.background_gradient(cmap=cm).set_precision(2))
        
    
if __name__=='__main__':
    main()



      
      