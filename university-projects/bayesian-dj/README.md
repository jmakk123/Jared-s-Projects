# Bayesian DJ

**Bayesian Machine Learning · MS-ADS · Winter 2026**
Team: Payton Stewart, Arthur Acker, Jared Maksoud
Original repo: https://github.com/paystewart/Bayesian-DJ

A music recommender that learns your taste in real time using Bayesian inference.

Most recommenders treat preferences as fixed point estimates. This one treats them as
probability distributions and updates them live. You describe a mood in plain English
("late night drive, something dark and smooth"), which is parsed into informative Gaussian
priors over nine Spotify audio features. Play and skip feedback refines those into a
posterior via **online Bayesian logistic regression with Laplace approximation**, and
**Thompson sampling** picks each next track, balancing exploration against exploitation.

### Running it

```bash
pip install -r requirements.txt

# Streamlit UI
streamlit run user_interface.py

# Interactive CLI
python -m bayesian_dj --prompt "chill indie, low energy, acoustic"

# Headless strategy comparison
python -m bayesian_dj --simulate --user-profile chill_listener --sim-rounds 50 --sim-repeats 20
```

### Data and credentials (not committed)

- **Spotify tracks dataset** — download and save as `kaggle_dataset.csv` in the project root.
  https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- **Spotify Web API** (optional: album art, previews, listening-history sync) — create an app
  at https://developer.spotify.com/dashboard and export:

  ```bash
  export SPOTIFY_CLIENT_ID=your_id
  export SPOTIFY_CLIENT_SECRET=your_secret
  ```

  Never commit these. The app runs without them, just without artwork and previews.

`README.upstream.md` is the original team README.
