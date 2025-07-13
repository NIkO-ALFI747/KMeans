namespace KMeans.Contracts;

public record KMeansResponse(float[][] Centroids, int[] Assignments);
