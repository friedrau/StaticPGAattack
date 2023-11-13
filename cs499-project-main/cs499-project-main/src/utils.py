import Levenshtein as lev

def calculate_cer(reference, hypothesis):
  """
  Calculate the character error rate (CER) between two strings.
  """

  # Compute the edit distance between the reference and the hypothesis
  edit_distance = lev.distance(reference, hypothesis)

  # Compute the character error rate
  cer = edit_distance / len(reference)

  return cer
