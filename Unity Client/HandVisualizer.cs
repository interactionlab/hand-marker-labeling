using UnityEngine;

/// <summary>
/// An example HandDataEvent Listener which receives the hand data and visualizes it.
/// Requires the HandTrackingClient to work.
/// </summary>
public class HandVisualizer : MonoBehaviour
{
	private GameObject[] spheresLeft = new GameObject[21];
	private GameObject[] spheresRight = new GameObject[21];
	
	
	/// <summary>
	/// Initializes the getHandDataEvent Listener and creates the game objects for data visualization.
	/// </summary>
	void Start()
	{
		HandTrackingClient.getHandDataEvent().AddListener(ReceiveData);
		for (int i = 0; i < spheresLeft.Length; i++)
		{
			spheresLeft[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
			spheresLeft[i].transform.localScale = new Vector3(0.005f, 0.005f, 0.005f);
			spheresLeft[i].transform.parent = gameObject.transform;
		}
		// second for loop is to keep the order of the game objects sorted by hand.
		for (int i = 0; i < spheresLeft.Length; i++)
		{
			spheresRight[i] = GameObject.CreatePrimitive(PrimitiveType.Sphere);
			spheresRight[i].transform.localScale = new Vector3(0.005f, 0.005f, 0.005f);
			spheresRight[i].transform.parent = gameObject.transform;
		}
	}

	
	private Hand latestHandDataLeft;
	private Hand latestHandDataRight;
	/// <summary>
	/// Receives the Hand data from the event and stores the data, so it will be visualized at the next update.
	/// </summary>
	/// <param name="hand"></param>
	private void ReceiveData(Hand hand)
	{
		if (hand.IsRightHand)
		{
			latestHandDataRight = hand;
		}
		else
		{
			latestHandDataLeft = hand;
		}
	}
	

	/// <summary>
	/// Visualizes the received Hand data.
	/// </summary>
	void Update () 
	{
		if (latestHandDataLeft != null)
		{
			for (int i = 0; i < spheresLeft.Length; i++)
			{
				spheresLeft[i].transform.position = latestHandDataLeft.getCoordinatesByIndex(i);
				spheresLeft[i].transform.name = latestHandDataLeft.Labels[i];
			}
		}
		if (latestHandDataRight != null)
		{
			for (int i = 0; i < spheresRight.Length; i++)
			{
				spheresRight[i].transform.position = latestHandDataRight.getCoordinatesByIndex(i);
				spheresRight[i].transform.name = latestHandDataRight.Labels[i];
			}
		}
	}
}
