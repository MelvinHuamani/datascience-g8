import requests
import streamlit as st

API_URL = "https://potato-mlops.onrender.com"

st.set_page_config(
    page_title="Diagnóstico de hojas de papa",
    page_icon="🥔",
    layout="centered",
)

st.title("Diagnóstico de enfermedades en hojas de papa")
st.write("Sube una imagen de una hoja para obtener una predicción del modelo.")

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    st.image(
        uploaded_file,
        caption="Imagen seleccionada",
        use_container_width=True,
    )

    if st.button("Analizar imagen", type="primary"):
        with st.spinner("Analizando la imagen..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "image/jpeg",
                        )
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                st.subheader("Resultados")

                cols = st.columns(3)

                for i, item in enumerate(result["top_3"]):
                    with cols[i]:
                        with st.container(border=True):
                            st.caption(f"Resultado {i + 1}")
                            st.markdown(f"**{item['class']}**")
                            st.metric(
                                label="Probabilidad",
                                value=f"{item['score']:.1%}",
                            )

            except requests.exceptions.RequestException as error:
                st.error(f"No se pudo completar la predicción: {error}")
            except (ValueError, KeyError, IndexError):
                st.error("La API respondió, pero no se pudo interpretar el resultado.")
