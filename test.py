import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
import io

# Streamlit app title
st.title("Customer Segmentation with K-means Clustering")

# File upload
uploaded_file = st.file_uploader("Upload your csv file", type=['csv'])

if uploaded_file is not None:
    # Read the Excel file
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # st.write(numeric_cols)
        columns= st.multiselect(
                    "Choose at least two numeric columns for clustering",
                    options=numeric_cols )
        
        X=df[columns].copy()   

        # User input for number of clusters
    
        # n_clusters = st.number_input("Select number of clusters", min_value=2, max_value=10)
        # st.write(X)

        cluster_options=st.radio("how do you want to choose the number of clusters",
                                 options=['let me choose', 'automatically detect'])
        if cluster_options=='let me choose':
            n_clusters=st.slider("Select number of clusters", min_value=2, max_value=10)
        else:
            n=[]
            for i in range(1,10):
                kmeans=KMeans(n_clusters=i)
                kmeans.fit(X)
                n.append(kmeans.inertia_)
            # Find optimal number of clusters using KneeLocator
            k_range = list(range(1, 10))
            knee = KneeLocator(k_range, n, curve='convex', direction='decreasing')
            n_clusters = knee.elbow   
            st.write("number of clusters selected is ", n_clusters)

            
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(X)
        # st.write(clusters)
        
        # Add cluster labels to the original dataframe
        df['Cluster'] = clusters
        
        # Display clustering results
        st.write("### Clustered Data Preview")
        st.dataframe(df.head())
        
        # Downloadable file
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        
        st.download_button(
            label="Download Clustered Data",
            data=output,
            file_name="clustered_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Display basic cluster statistics
        st.write("### Cluster Statistics")
        stats = df.groupby('Cluster')[numeric_cols].mean()
        st.write(stats)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please upload an Excel file to proceed.")