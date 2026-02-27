using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

public class UDPReceiver : MonoBehaviour
{
    Thread receiveThread;
    UdpClient client;
    public int port = 5005; // Debe coincidir con el puerto de Python
    public float sensitivity = 0.01f; // Ajustar para la velocidad de movimiento

    // Variables para almacenar la posición recibida
    private float targetX = 0;
    private float targetY = 0;
    private bool newData = false;

    // Referencias para mapear coordenadas de cámara a Unity
    // Asumiendo resolución de cámara 640x480 (ajustar si es diferente)
    private float camWidth = 640f;
    private float camHeight = 480f;

    void Start()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log("UDP Receiver started on port " + port);
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);

                // Formato esperado: "x,y"
                string[] parts = text.Split(',');
                if (parts.Length == 2)
                {
                    float x = float.Parse(parts[0]);
                    float y = float.Parse(parts[1]);

                    // Mapear coordenadas de pantalla (0,0 arriba-izq) a Unity
                    // Unity: x (-5 a 5), y (-3 a 3) aprox, dependiendo de la cámara
                    
                    // Invertir Y porque en OpenCV 0 está arriba y en Unity abajo suele ser negativo
                    // Normalizar de 0..640 a -1..1
                    float normX = (x / camWidth) * 2 - 1; 
                    float normY = -((y / camHeight) * 2 - 1); 

                    // Escalar al mundo de Unity (ej. ancho de 10 unidades)
                    targetX = normX * 10f; 
                    targetY = normY * 5f; 
                    
                    newData = true;
                }
            }
            catch (System.Exception err)
            {
                Debug.Log(err.ToString());
            }
        }
    }

    void Update()
    {
        if (newData)
        {
            // Mover suavemente el objeto a la nueva posición
            Vector3 newPos = new Vector3(targetX, targetY, transform.position.z);
            transform.position = Vector3.Lerp(transform.position, newPos, Time.deltaTime * 5f);
            newData = false;
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null && receiveThread.IsAlive)
            receiveThread.Abort();

        if (client != null)
            client.Close();
    }
}
