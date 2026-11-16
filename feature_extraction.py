
import pandas as pd
import re
from urllib.parse import urlparse

def extract_features(url):

    url = str(url).strip()

    # Default values for malformed URLs
    hostname = ""
    path = ""
    scheme = ""

    try:
        # Safely parse URL
        safe_url = url if "://" in url else "http://" + url
        parsed = urlparse(safe_url)

        hostname = parsed.hostname or ""
        path = parsed.path.lower()
        scheme = parsed.scheme.lower()

    except (ValueError, TypeError):
        pass

    features = {}

    features["url_length"] = len(url)
    features["hostname_length"] = len(hostname)
    features["path_length"] = len(path)

    features["dot_count"] = url.count(".")
    features["hyphen_count"] = url.count("-")
    features["at_count"] = url.count("@")
    features["question_count"] = url.count("?")
    features["equal_count"] = url.count("=")

    features["digit_count"] = sum(
        c.isdigit() for c in url
    )

    features["special_char_count"] = len(
        re.findall(r"[^a-zA-Z0-9]", url)
    )

    features["https"] = int(scheme == "https")

    features["ip_address"] = int(
        bool(re.fullmatch(
            r"\d{1,3}(?:\.\d{1,3}){3}",
            hostname
        ))
    )

    features["subdomain_count"] = max(
        0, len(hostname.split(".")) - 2
    ) if hostname else 0

    suspicious_words = [
        "login", "verify", "account", "update",
        "secure", "banking", "password", "signin",
        "confirm", "wallet"
    ]

    features["suspicious_word_count"] = sum(
        word in url.lower()
        for word in suspicious_words
    )

    shortening_services = [
        "bit.ly", "tinyurl.com", "t.co",
        "goo.gl", "ow.ly", "is.gd"
    ]

    features["shortening_service"] = int(
        any(service in hostname for service in shortening_services)
    )

    return features


if __name__ == "__main__":

    df = pd.read_csv("dataset/cleaned_urls.csv")

    print("Extracting URL features...")

    feature_data = pd.DataFrame(
        df["url"].apply(extract_features).tolist()
    )

    feature_data["type"] = df["type"].values

    feature_data.to_csv(
        "dataset/extracted_features.csv",
        index=False
    )

    print("\nFeature extraction completed!")
    print("Total records:", len(feature_data))
    print("Total features:", len(feature_data.columns) - 1)
    print("Saved to dataset/extracted_features.csv")
