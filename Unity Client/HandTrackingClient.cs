using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEditor;
using UnityEngine;
using UnityEngine.Events;

/// <summary>
/// The HandTrackingClient receives the labeled Hand data of the python LabelingClient and provides it via the
/// the HandDataEvent. Create a new data Listener like:
/// HandTrackingClient.getHandDataEvent().AddListener(ReceiveData);
/// </summary>
public class HandTrackingClient : MonoBehaviour {
	private const int listenPort = 1512;  
	private static Thread receiveThread = null;
	private static bool done = false;
	
	
	/// <summary>
	/// Sets whether the unity or motive axis are used for the data.
	/// </summary>
	public static bool useUnityCoordinateSystem = true;
	
	private static HandDataEvent _handDataEvent;

	
	/// <summary>
	/// Returns the HandDataEvent, if it is not created yet it will be initialized together with the data receiver thread.
	/// </summary>
	/// <returns></returns>
	public static HandDataEvent getHandDataEvent()
	{
		if (receiveThread == null)
		{
			Debug.Log("Starting HandTrackingClient");
			done = false;
			receiveThread = new Thread(Listen);
			receiveThread.IsBackground = true;
			receiveThread.Start();
			_handDataEvent = new HandDataEvent();
		}

		return _handDataEvent;
	}
	
	
	/// <summary>
	/// The Event which provides the labeled Hand data whenever it is received.
	/// Initialize a Listener like:
	/// HandTrackingClient.getHandDataEvent().AddListener(ReceiveData);
	/// </summary>
	[System.Serializable]
	public class HandDataEvent : UnityEvent<Hand> { }
	
	// initializes the HandDataEvent as well as the receiveThread at startup.
	void Start ()
	{
		getHandDataEvent();
	}
	
	// Destroys the HandDataEvent and the receiveThread.
	void OnDestroy()
	{
		done = true;
		receiveThread = null;
		_handDataEvent = null;
	}
	

	/// <summary>
	/// Listens for the labeled Hand data at port 1512.
	/// Whenever data is received a Hand object is created and the _handDataEvent is invoked.
	/// </summary>
	private static void Listen()   
	{  
		UdpClient listener = new UdpClient(listenPort);  
		IPEndPoint groupEP = new IPEndPoint(IPAddress.Any,listenPort);  

		try   
		{  
			Debug.Log("HandTrackingClient is waiting for Data on port " + listenPort);  
			while (!done)   
			{  
				byte[] bytes = listener.Receive( ref groupEP);
				Boolean isRightHand = BitConverter.ToSingle(bytes, 0).Equals(1.0f);
				
				int offset = 4;
				List<float> values = new List<float>();
				while (offset < bytes.Length)
				{
					values.Add(BitConverter.ToSingle(bytes, offset));
					offset += 4;
				}
				_handDataEvent.Invoke(new Hand(values, isRightHand));
			}
		}   
		catch (Exception e)   
		{  
			Console.WriteLine(e.ToString());  
		}  
		finally  
		{  
			listener.Close();  
		}  
	}
}


/// <summary>
/// Contains the the received marker data as well as the marker labels.
/// A new Hand object will be created whenever data is received from the python LabelingClient.
/// </summary>
public class Hand
{
	// In this order the data will be given to the constructor.
	public readonly String[] Labels =
	{
		"Thumb_Fn", "Thumb_DIP", "Thumb_PIP", "Thumb_MCP",
		"Index_Fn", "Index_DIP", "Index_PIP", "Index_MCP",
		"Middle_Fn", "Middle_DIP", "Middle_PIP", "Middle_MCP",
		"Ring_Fn", "Ring_DIP", "Ring_PIP", "Ring_MCP",
		"Little_Fn", "Little_DIP", "Little_PIP", "Little_MCP",
		"Wrist"
	};
	
	
	/// <summary>
	/// Contains the coordinates of the markers in the order of Labels
	/// </summary>
	public List<Vector3> Coordinates = new List<Vector3>();
	public Boolean IsRightHand;
	
	
	/// <summary>
	/// Creates a Hand object with the in values given marker data.
	/// </summary>
	/// <param name="values"></param> the marker coordinates in x y z order.
	/// <param name="isRightHand"></param> whether the hand is left or right.
	public Hand(List<float> values, Boolean isRightHand)
	{
		IsRightHand = isRightHand;
		for (int i = 0; i + 2 < values.Count; i += 3)
		{
			Vector3 coords;
			if (HandTrackingClient.useUnityCoordinateSystem)
			{
				coords = new Vector3(-values[i], values[i + 1], values[i + 2]);
			}
			else
			{
				coords = new Vector3(values[i], values[i + 1], values[i + 2]);
			}
			Coordinates.Add(coords);
		}
	}

	public int getNumberOfMarkers()
	{
		return Coordinates.Count;
	}

	
	/// <summary>
	/// Returns the coordinates of the requested marker, or (0,0,0) if the marker is not in the data.
	/// </summary>
	/// <param name="label"></param>
	/// <returns></returns>
	public Vector3 getCoordinatesByLabel(String label)
	{
		int index = getIndexByLabel(label);
		if (index >= 0 && index < getNumberOfMarkers())
		{
			return Coordinates[index];
		}

		return new Vector3(0,0,0);
	}

	
	/// <summary>
	/// Returns the coordinates of the requested marker, or (0,0,0) if the index is out of bounds.
	/// </summary>
	/// <param name="label"></param>
	/// <returns></returns>
	public Vector3 getCoordinatesByIndex(int index)
	{
		if (index < getNumberOfMarkers())
		{
			return Coordinates[index];
		}

		return new Vector3(0,0,0);
	}

	
	/// <summary>
	/// Returns the index of a marker in the Coordinates list by its label.
	/// </summary>
	/// <param name="label"></param>
	/// <returns></returns>
	public int getIndexByLabel(String label)
	{
		return Array.IndexOf(Labels, label);
	}
}
